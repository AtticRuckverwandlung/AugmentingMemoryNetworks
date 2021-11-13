import abc
import torch

from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, AnyStr


class JointModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    labels: Optional[torch.Tensor] = None


class Encoder(BertPreTrainedModel):
    """
    Base transformer encoder class
    """
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = None

        self.bert = BertModel(config)  # Must keep this name as 'bert' to ensure weights are loaded properly
        self.bert2 = BertModel(config)
        self.num_factoids = 4
        self.evaluate_responses = False

        self.init_weights()

    def index_candidates(self, samples):
        # Given a dataset, we want to build a FAISS index
        with torch.no_grad():
            embedding = self.get_embedding(
                torch.tensor(samples["y_input_ids"]).to(self.device),
                torch.tensor(samples["y_attention_mask"]).to(self.device)
            ).cpu().numpy()
            samples["embedding"] = embedding

        return samples

    def prepare_candidates(
            self,
            y_input_ids: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None
    ):
        ground_truth_input_ids = y_input_ids.unsqueeze(1)
        ground_truth_attention_mask = y_attention_mask.unsqueeze(1)
        combined_input_ids = torch.cat([ground_truth_input_ids, candidate_input_ids], 1)
        combined_attention_mask = torch.cat([ground_truth_attention_mask, candidate_attention_mask], 1)

        return combined_input_ids, combined_attention_mask

    def score_factoids(
            self,
            y_input_ids: Optional[torch.Tensor] = None,
            z_input_ids: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            query_embed: Optional[torch.Tensor] = None,
            factoid_embed: Optional[torch.Tensor] = None,
            batch_negatives=False  # Whether to include factoids from other samples as negatives
    ) -> torch.IntTensor:

        if y_input_ids is None and query_embed is None:
            raise ValueError("Must specify either input_ids or query_embed")
        if z_input_ids is None and factoid_embed is None:
            raise ValueError("Must specify either input_ids or factoid_embed")

        bsz = y_input_ids.shape[0] if y_input_ids is not None else query_embed.shape[0]
        if lengths is not None:
            num_factoids = torch.max(lengths)
        elif factoid_embed is not None:
            num_factoids = factoid_embed.shape[0] // bsz
        else:
            num_factoids = z_input_ids.shape[1]

        if query_embed is None:
            query_embed = self._get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
        elif factoid_embed is None:
            z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape(
                [bsz * num_factoids, -1])

            factoid_embed = self.get_embedding(input_ids=z_input_ids, attention_mask=z_attention_mask)

        if batch_negatives:
            scores = torch.matmul(query_embed, factoid_embed.T)
        else:
            factoid_embed = factoid_embed.reshape([bsz, num_factoids, -1])

            # response_embed has shape B x D, factoid_embed has shape B x F x D
            # We want to score each response against its respective factoids (but not the negatives from other samples)
            query_embed = query_embed.unsqueeze(1).expand(-1, num_factoids, -1)
            scores = torch.multiply(factoid_embed, query_embed).sum(-1)  # B x F

        return scores

    def get_embedding_context(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        out = self.bert2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state[:, 0, :]

        return out

    def get_embedding(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state[:, 0, :]

        return out

    @abc.abstractmethod
    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            z_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:
        """
        Abstract forward method for grounded models that implement BERT-like encoder.
        :param x_input_ids: 2D tensor of shape batch size x length containing dialogue contexts
        :param y_input_ids: 2D tensor of shape batch size x length containing dialogue responses
        :param z_input_ids: 3D tensor of shape batch size x num factoids x length containing grounded factoids
        :param x_attention_mask: 2D tensor of shape batch size x length
        :param y_attention_mask: 2D tensor of shape batch size x length
        :param z_attention_mask: 3D tensor of shape batch size x num factoids x length
        :param compute_loss: boolean whether or not loss should be calculated (e.g. during training and evaluation)
        :param candidate_input_ids: 3D tensor of shape batch size x num candidates x length
        :param candidate_attention_mask: 3D tensor of shape batch size x num candidates x length
        :param lengths: 1D tensor of shape batch size
        :return: outputs
        """
        pass

    def get_subset_factoids(
            self,
            query_input_ids: Optional[torch.Tensor] = None,
            query_attention_mask: Optional[torch.Tensor] = None,
            query_embed: Optional[torch.Tensor] = None,
            z_input_ids: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            num_factoids: Optional[torch.Tensor] = None,
            query_type_ids: Optional[torch.Tensor] = None
    ):
        assert query_input_ids is not None or query_embed is not None

        device = query_input_ids.device if query_input_ids is not None else query_embed.device
        bsz = query_input_ids.shape[0] if query_input_ids is not None else query_embed.shape[0]

        k = self.num_factoids
        z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
        z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz * num_factoids, -1])

        if query_embed is None:
            context_embed = self.get_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                token_type_ids=query_type_ids
            )
        else:
            context_embed = query_embed

        factoid_chunks = []
        step = (bsz * num_factoids) // 8
        for i in range(0, bsz * num_factoids, step):
            factoid_embed = self.get_embedding(z_input_ids[i:i + step], z_attention_mask[i:i + step])
            factoid_chunks.append(factoid_embed)
        factoid_embed = torch.cat(factoid_chunks, dim=0)
        knowledge_scores = self.score_factoids(query_embed=context_embed, factoid_embed=factoid_embed)
        _, top_k_indices = torch.topk(knowledge_scores, dim=-1, k=k)
        top_k_indices = top_k_indices.cpu().numpy()
        shortlist_z_input_ids = []
        shortlist_z_attention_mask = []
        z_input_ids = z_input_ids.view([bsz, num_factoids, -1])
        z_attention_mask = z_attention_mask.view([bsz, num_factoids, -1])
        for i in range(bsz):
            shortlist_z_input_ids.append(
                z_input_ids[i].index_select(0, torch.tensor(top_k_indices[i]).to(device)))
            shortlist_z_attention_mask.append(
                z_attention_mask[i].index_select(0, torch.tensor(top_k_indices[i]).to(device)))
        z_input_ids = torch.stack(shortlist_z_input_ids, dim=0).view([bsz, k, -1])
        z_attention_mask = torch.stack(shortlist_z_attention_mask, dim=0).view([bsz, k, -1])

        return {
            "z_input_ids": z_input_ids,
            "z_attention_mask": z_attention_mask
        }


class RetrievalAugmentedRetrieval(Encoder):

    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            z_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:

        device = x_input_ids.device
        bsz = x_input_ids.shape[0]

        num_factoids = torch.max(lengths)

        if self.training:
            with torch.no_grad():
                subset_factoids = self.get_subset_factoids(
                    query_input_ids=x_input_ids,
                    query_attention_mask=x_attention_mask,
                    z_input_ids=z_input_ids,
                    z_attention_mask=z_attention_mask,
                    num_factoids=num_factoids
                )
                num_factoids = self.num_factoids

            z_input_ids, z_attention_mask = subset_factoids["z_input_ids"], subset_factoids["z_attention_mask"]
        else:
            z_input_ids = z_input_ids[:, :num_factoids, :]
            z_attention_mask = z_attention_mask[:, :num_factoids, :]

        context_embed = self.get_embedding_context(x_input_ids, x_attention_mask)

        # Scores has shape b x k
        factoid_scores = self.score_factoids(
            query_embed=context_embed,
            z_input_ids=z_input_ids,
            z_attention_mask=z_attention_mask
        )

        # Compute posterior scores for individual <context, factoid> pairs
        broadcast_x_input_ids = x_input_ids.unsqueeze(1).expand(-1, num_factoids, -1)
        broadcast_x_attention_mask = x_attention_mask.unsqueeze(1).expand(-1, num_factoids, -1)

        combined_input_ids = torch.cat([broadcast_x_input_ids, z_input_ids], -1)
        combined_attention_mask = torch.cat([broadcast_x_attention_mask, z_attention_mask], -1)
        token_type_ids = torch.cat([
            torch.zeros(broadcast_x_input_ids.shape, dtype=torch.long),
            torch.ones(z_input_ids.shape, dtype=torch.long)
        ], dim=-1).to(device)

        # embed has shape b x k x d
        combined_input_ids = combined_input_ids.reshape([bsz*num_factoids, -1])
        combined_attention_mask = combined_attention_mask.reshape([bsz*num_factoids, -1])
        token_type_ids = token_type_ids.reshape([bsz*num_factoids, -1])
        combined_embed = self.get_embedding_context(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            token_type_ids=token_type_ids
        ).reshape([bsz, num_factoids, -1])

        # Marginalisation step
        softmax = F.softmax(factoid_scores, 1)
        marginalised_embedding = torch.einsum("bk, bkd -> bkd", softmax, combined_embed).sum(1)

        if self.training:
            response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
            scores = torch.matmul(marginalised_embedding, response_embed.T)
            labels = torch.arange(bsz).to(device)

        else:
            scores = factoid_scores

            if self.evaluate_responses:
                input_ids, attention_mask = self.prepare_candidates(
                    y_input_ids=y_input_ids,
                    y_attention_mask=y_attention_mask,
                    candidate_input_ids=candidate_input_ids,
                    candidate_attention_mask=candidate_attention_mask
                )
                scores = self.score_factoids(
                    query_embed=marginalised_embedding,
                    z_input_ids=input_ids,
                    z_attention_mask=attention_mask
                )
            labels = torch.zeros(bsz, dtype=torch.long).to(device)

        loss = torch.nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(loss=loss, logits=scores)


class ConcatTransformer(Encoder):
    """
    Retrieval transformer that simply concatenates persona into context
    """
    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            z_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:

        bsz = x_input_ids.shape[0]
        device = x_input_ids.device
        num_factoids = torch.max(lengths)

        # Unroll persona
        z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz, -1])  # bsz x k x words -> bsz x (k * words)
        z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz, -1])
        token_type_ids = torch.cat([
            torch.zeros(z_input_ids.size(), dtype=torch.long),
            torch.ones(x_input_ids.size(), dtype=torch.long)
        ], dim=-1).to(device)
        combined_input_ids = torch.cat([z_input_ids, x_input_ids], dim=-1)
        combined_attention_mask = torch.cat([z_attention_mask, x_attention_mask], dim=-1)
        context_embed = self.get_embedding_context(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            token_type_ids=token_type_ids
        )
        if self.training:
            response_embed = self.get_embedding(y_input_ids, y_attention_mask)
            labels = torch.arange(bsz).to(device)
            scores = torch.matmul(context_embed, response_embed.T)
        else:
            input_ids, attention_mask = self.prepare_candidates(
                y_input_ids=y_input_ids,
                y_attention_mask=y_attention_mask,
                candidate_input_ids=candidate_input_ids,
                candidate_attention_mask=candidate_attention_mask
            )
            labels = torch.zeros(bsz, dtype=torch.long).to(device)
            scores = self.score_factoids(
                query_embed=context_embed,
                z_input_ids=input_ids,
                z_attention_mask=attention_mask
            )
        loss = nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(logits=scores, loss=loss)


class BiEncoder(Encoder):
    """
    No knowledge bi-encoder retriever
    """
    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:

        bsz = x_input_ids.shape[0]
        device = x_input_ids.device

        context_embed = self.get_embedding_context(x_input_ids, x_attention_mask)
        if self.training:
            response_embed = self.get_embedding(y_input_ids, y_attention_mask)
            labels = torch.arange(bsz).to(device)
            scores = torch.matmul(context_embed, response_embed.T)
        else:
            input_ids, attention_mask = self.prepare_candidates(
                y_input_ids=y_input_ids,
                y_attention_mask=y_attention_mask,
                candidate_input_ids=candidate_input_ids,
                candidate_attention_mask=candidate_attention_mask
            )
            labels = torch.zeros(bsz, dtype=torch.long).to(device)
            scores = self.score_factoids(
                query_embed=context_embed,
                z_input_ids=input_ids,
                z_attention_mask=attention_mask
            )
        loss = nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(logits=scores, loss=loss)


class MemNet(Encoder):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_factoids = 4
        self.bert2 = BertModel(config)
        self.faiss_dataset = None
        self.evaluate_responses = False

    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            z_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:

        device = x_input_ids.device
        bsz = x_input_ids.shape[0]

        if self.training:

            num_factoids = torch.max(lengths)

            if self.faiss_dataset is None:
                with torch.no_grad():
                    subset_factoids = self.get_subset_factoids(
                        query_input_ids=x_input_ids,
                        query_attention_mask=x_attention_mask,
                        z_input_ids=z_input_ids,
                        z_attention_mask=z_attention_mask,
                        num_factoids=num_factoids
                    )
                z_input_ids, z_attention_mask = subset_factoids["z_input_ids"], subset_factoids["z_attention_mask"]
                context_embed = self.get_embedding_context(x_input_ids, x_attention_mask)

            # Retrieve relevant factoids from asynchronous FAISS index
            else:
                context_embed = self.get_embedding_context(x_input_ids, x_attention_mask)
                scores, examples = self.faiss_dataset.get_nearest_examples_batch("embedding", context_embed.detach().cpu().numpy(),
                                                                                 k=self.num_factoids)
                z_input_ids = torch.tensor([k for ks in examples for k in ks["z_input_ids"]]).to(device).view(
                    [bsz * self.num_factoids, -1])
                z_attention_mask = torch.tensor([k for ks in examples for k in ks["z_attention_mask"]]).to(device).view(
                    [bsz * self.num_factoids, -1])

            factoid_embed = self.get_embedding(
                z_input_ids.view([bsz * self.num_factoids, -1]),
                z_attention_mask.view([bsz * self.num_factoids, -1])
            )

            prior_scores = self.score_factoids(query_embed=context_embed, factoid_embed=factoid_embed)
            softmax = F.softmax(prior_scores, -1)

            values = torch.einsum("bf, bfd -> bfd", softmax, factoid_embed.view([bsz, self.num_factoids, -1])).sum(1)
            final_embedding = values + context_embed

            response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)

            scores = torch.matmul(final_embedding, response_embed.T)
            labels = torch.arange(bsz).to(device)
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        else:
            context_embed = self.get_embedding_context(x_input_ids, x_attention_mask)
            num_factoids = torch.max(lengths)
            z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            factoid_embed = self.get_embedding(z_input_ids, z_attention_mask)

            scores = self.score_factoids(query_embed=context_embed, factoid_embed=factoid_embed)
            labels = torch.zeros(bsz, dtype=torch.long).to(device)

            if self.evaluate_responses:
                softmax = F.softmax(scores, -1)
                values = torch.einsum("bf, bfd -> bfd", softmax, factoid_embed.view([bsz, num_factoids, -1])).sum(
                    1)
                final_embedding = values + context_embed

                input_ids, attention_mask = self.prepare_candidates(
                    y_input_ids=y_input_ids,
                    y_attention_mask=y_attention_mask,
                    candidate_input_ids=candidate_input_ids,
                    candidate_attention_mask=candidate_attention_mask
                )
                scores = self.score_factoids(
                    query_embed=final_embedding,
                    z_input_ids=input_ids,
                    z_attention_mask=attention_mask
                )

            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(loss=loss, logits=scores)


class PolyMemNet(Encoder):

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_factoids = 4
        self.response_cache = None
        self.bert2 = BertModel(config)

        self.num_codes = 128
        self.codes = nn.Embedding(self.num_codes, config.hidden_size)

        self.evaluate_responses = False
        self.faiss_dataset = None

        self.use_mha = False
        if self.use_mha:
            self.mha = nn.MultiheadAttention(config.hidden_size, num_heads=config.num_attention_heads)

    def get_subset_factoids(
            self,
            query_input_ids: Optional[torch.Tensor] = None,
            query_attention_mask: Optional[torch.Tensor] = None,
            query_embed: Optional[torch.Tensor] = None,
            z_input_ids: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            num_factoids: Optional[torch.Tensor] = None,
            query_type_ids: Optional[torch.Tensor] = None
    ):
        assert query_input_ids is not None or query_embed is not None

        device = query_input_ids.device if query_input_ids is not None else query_embed.device
        bsz = query_input_ids.shape[0] if query_input_ids is not None else query_embed.shape[0]

        k = self.num_factoids

        context_hidden_states = self.bert2(input_ids=query_input_ids,
                                          attention_mask=query_attention_mask).last_hidden_state  # bsz x words x dim
        context_codes = self.codes(torch.arange(self.num_codes).to(device)).unsqueeze(0).expand(bsz, -1,
                                                                                                -1)  # bsz x codes x dim
        context_values = self.dot_product_attention(context_codes, context_hidden_states)

        z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
        z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
        factoid_chunks = []
        step = (bsz * num_factoids) // 8
        for i in range(0, bsz * num_factoids, step):
            factoid_embed = self.get_embedding(z_input_ids[i:i + step], z_attention_mask[i:i + step])
            factoid_chunks.append(factoid_embed)
        factoid_embed = torch.cat(factoid_chunks, dim=0)
        factoid_embed = factoid_embed.reshape([bsz, num_factoids, -1])

        factoid_values, scores = self.dot_product_attention(context_values, factoid_embed, return_scores=True)
        scores = scores.mean(1)  # Average across codes to obtain bsz x num_factoids matrix

        _, top_k_indices = torch.topk(scores, dim=-1, k=k)
        top_k_indices = top_k_indices.cpu().numpy()
        shortlist_z_input_ids = []
        shortlist_z_attention_mask = []
        z_input_ids = z_input_ids.view([bsz, num_factoids, -1])
        z_attention_mask = z_attention_mask.view([bsz, num_factoids, -1])
        for i in range(bsz):
            shortlist_z_input_ids.append(
                z_input_ids[i].index_select(0, torch.tensor(top_k_indices[i]).to(device)))
            shortlist_z_attention_mask.append(
                z_attention_mask[i].index_select(0, torch.tensor(top_k_indices[i]).to(device)))
        z_input_ids = torch.stack(shortlist_z_input_ids, dim=0).view([bsz, k, -1])
        z_attention_mask = torch.stack(shortlist_z_attention_mask, dim=0).view([bsz, k, -1])

        return {
            "z_input_ids": z_input_ids,
            "z_attention_mask": z_attention_mask
        }

    def dot_product_attention(self, query, key, return_scores=False):
        if self.use_mha:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            values, scores = self.mha(query, key, key)
        else:
            key_t = key.transpose(2, 1)
            scores = torch.einsum("bqd, bdk -> bqk", query, key_t)
            softmax = torch.softmax(scores, -1)
            values = torch.einsum("bqk, bkd -> bqd", softmax, key)

        if return_scores:
            return values, scores

        return values

    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            z_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            z_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            **kwargs
    ) -> JointModelOutput:

        device = x_input_ids.device
        bsz = x_input_ids.shape[0]

        context_hidden_states = self.bert2(input_ids=x_input_ids,
                                          attention_mask=x_attention_mask).last_hidden_state  # bsz x words x dim
        context_codes = self.codes(torch.arange(self.num_codes).to(device)).unsqueeze(0).expand(bsz, -1, -1)  # bsz x codes x dim
        context_values = self.dot_product_attention(context_codes, context_hidden_states)

        num_factoids = torch.max(lengths)

        if self.training:
            with torch.no_grad():
                subset_factoids = self.get_subset_factoids(
                    query_input_ids=x_input_ids,
                    query_attention_mask=x_attention_mask,
                    z_input_ids=z_input_ids,
                    z_attention_mask=z_attention_mask,
                    num_factoids=num_factoids
                )
            z_input_ids, z_attention_mask = subset_factoids["z_input_ids"], subset_factoids["z_attention_mask"]

            z_input_ids = z_input_ids[:, :self.num_factoids, :].reshape([bsz * self.num_factoids, -1])
            z_attention_mask = z_attention_mask[:, :self.num_factoids, :].reshape([bsz * self.num_factoids, -1])
            factoid_embed = self.get_embedding(z_input_ids, z_attention_mask)  # bsz*num_factoids x dim
            factoid_embed = factoid_embed.view([bsz, self.num_factoids, -1])  # bsz x num_factoids x dim
            factoid_values = self.dot_product_attention(context_values, factoid_embed)
            final_embedding = context_values + factoid_values
            response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
            response_embed = response_embed.unsqueeze(0).expand(bsz, -1, -1)
            response_values = self.dot_product_attention(response_embed, final_embedding)
            scores = (response_values * response_embed).sum(-1)  # bsz x bsz
            labels = torch.arange(bsz).to(device)
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        else:
            z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            factoid_embed = self.get_embedding(z_input_ids, z_attention_mask)  # bsz*num_factoids x dim
            factoid_embed = factoid_embed.view([bsz, num_factoids, -1])  # bsz x num_factoids x dim

            if not self.evaluate_responses:
                factoid_values, scores = self.dot_product_attention(context_values, factoid_embed, return_scores=True)
                scores = scores.mean(1)  # Average across codes to obtain bsz x num_factoids matrix
            else:
                factoid_values = self.dot_product_attention(context_values, factoid_embed)
                final_embedding = context_values + factoid_values

                # For testing accuracy we evaluate against response candidates
                if self.faiss_dataset is None:
                    input_ids, attention_mask = self.prepare_candidates(
                        y_input_ids=y_input_ids,
                        y_attention_mask=y_attention_mask,
                        candidate_input_ids=candidate_input_ids,
                        candidate_attention_mask=candidate_attention_mask
                    )
                    num_cands = candidate_input_ids.shape[1] + 1
                    input_ids = input_ids.reshape([bsz * num_cands, -1])
                    attention_mask = attention_mask.reshape([bsz * num_cands, -1])

                    response_embed = self.get_embedding(input_ids=input_ids, attention_mask=attention_mask).reshape([bsz, num_cands, -1])

                    response_values = self.dot_product_attention(response_embed, final_embedding)
                    scores = (response_values * response_embed).sum(-1)

                # For generating predictions
                else:
                    k = 5
                    query = final_embedding.reshape([bsz*self.num_codes, -1]).cpu().numpy()
                    _, examples = self.faiss_dataset.get_nearest_examples_batch("embedding", query, k=k)
                    y_input_ids = torch.tensor([y for pair in examples for y in pair["y_input_ids"]]).to(device)
                    y_attention_mask = torch.tensor([y for pair in examples for y in pair["y_attention_mask"]]).to(
                        device)  # bsz * num_codes * 10 x length
                    response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
                    response_embed = response_embed.reshape([bsz, self.num_codes*k, -1])

                    response_values = self.dot_product_attention(response_embed, final_embedding)
                    scores = (response_values * response_embed).sum(-1)
                    pred = scores.argmax(1)

                    return y_input_ids.reshape([bsz, self.num_codes*k, -1]), pred

            labels = torch.zeros(bsz, dtype=torch.long).to(device)
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(loss=loss, logits=scores)


class Polyencoder(PolyMemNet):
    """
    No knowledge version of the polyencoder
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.evaluate_responses = True  # No knowledge, so this is only form of evaluation


    def forward(
            self,
            x_input_ids: torch.Tensor,
            y_input_ids: torch.Tensor,
            x_attention_mask: Optional[torch.Tensor] = None,
            y_attention_mask: Optional[torch.Tensor] = None,
            compute_loss: Optional[bool] = True,
            candidate_input_ids: Optional[torch.Tensor] = None,
            candidate_attention_mask: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
    ) -> JointModelOutput:

        device = x_input_ids.device
        bsz = x_input_ids.shape[0]

        context_hidden_states = self.bert2(input_ids=x_input_ids,
                                          attention_mask=x_attention_mask).last_hidden_state  # bsz x words x dim
        context_codes = self.codes(torch.arange(self.num_codes).to(device)).unsqueeze(0).expand(bsz, -1,
                                                                                                -1)  # bsz x codes x dim
        context_values = self.dot_product_attention(context_codes, context_hidden_states)

        if self.training:
            final_embedding = context_values
            response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
            response_embed = response_embed.unsqueeze(0).expand(bsz, -1, -1)
            response_values = self.dot_product_attention(response_embed, final_embedding)
            scores = (response_values * response_embed).sum(-1)  # bsz x bsz
            labels = torch.arange(bsz).to(device)
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        else:
            final_embedding = context_values

            # For testing accuracy we evaluate against response candidates
            if self.faiss_dataset is None:
                input_ids, attention_mask = self.prepare_candidates(
                    y_input_ids=y_input_ids,
                    y_attention_mask=y_attention_mask,
                    candidate_input_ids=candidate_input_ids,
                    candidate_attention_mask=candidate_attention_mask
                )
                num_cands = candidate_input_ids.shape[1] + 1
                input_ids = input_ids.reshape([bsz * num_cands, -1])
                attention_mask = attention_mask.reshape([bsz * num_cands, -1])
                response_embed = self.get_embedding(input_ids=input_ids, attention_mask=attention_mask).reshape(
                    [bsz, num_cands, -1])

                response_values = self.dot_product_attention(response_embed, final_embedding)
                scores = (response_values * response_embed).sum(-1)

            # For measuring F1-score
            else:
                k = 20
                query = final_embedding.reshape([bsz * self.num_codes, -1]).cpu().numpy()
                _, examples = self.faiss_dataset.get_nearest_examples_batch("embedding", query, k=k)
                y_input_ids = torch.tensor([y for pair in examples for y in pair["y_input_ids"]]).to(device)
                y_attention_mask = torch.tensor([y for pair in examples for y in pair["y_attention_mask"]]).to(
                    device)  # bsz * num_codes * 10 x length
                response_embed = self.get_embedding(input_ids=y_input_ids, attention_mask=y_attention_mask)
                response_embed = response_embed.reshape([bsz, self.num_codes * k, -1])

                response_values = self.dot_product_attention(response_embed, final_embedding)
                scores = (response_values * response_embed).sum(-1)
                pred = scores.argmax(1)

                return y_input_ids.reshape([bsz, self.num_codes * k, -1]), pred

            labels = torch.zeros(bsz, dtype=torch.long).to(device)
            loss = torch.nn.CrossEntropyLoss()(scores, labels)

        return JointModelOutput(loss=loss, logits=scores)


class SupervisedLearning(Encoder):
    """
    Simply supervised approach that maps contexts and factoids into shared latent space.
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_factoids = 4

    def forward(
            self,
            x_input_ids: torch.IntTensor,
            y_input_ids: torch.IntTensor,
            z_input_ids: torch.IntTensor,
            x_attention_mask: Optional[torch.IntTensor] = None,
            y_attention_mask: Optional[torch.IntTensor] = None,
            z_attention_mask: Optional[torch.IntTensor] = None,
            lengths: Optional[torch.IntTensor] = None,
            **kwargs

    ) -> JointModelOutput:

        device = x_input_ids.device
        bsz = x_input_ids.shape[0]
        if lengths is None:
            lengths = torch.ones(bsz, dtype=torch.int)

        if self.training:
            # During training, we randomly select k factoids from the batch, as there are too many to compute (c.61 per sample)
            num_factoids = min(z_input_ids.shape[1], self.num_factoids)
            indices = torch.cat([torch.zeros(1, dtype=torch.int), torch.randint(1, z_input_ids.shape[1], (num_factoids-1,))], 0).to(device)
            z_input_ids = z_input_ids.index_select(1, indices).reshape([bsz * num_factoids, -1])
            z_attention_mask = z_attention_mask.index_select(1, indices).reshape([bsz * num_factoids, -1])
        else:
            num_factoids = torch.max(lengths)
            z_input_ids = z_input_ids[:, :num_factoids, :].reshape([bsz * num_factoids, -1])
            z_attention_mask = z_attention_mask[:, :num_factoids, :].reshape([bsz * num_factoids, -1])

        context_embed = self.get_embedding_context(input_ids=x_input_ids, attention_mask=x_attention_mask)
        factoid_embed = self.get_embedding(z_input_ids, z_attention_mask)

        if self.training:
            scores = torch.matmul(context_embed, factoid_embed.T)
            # targets should be [0, max_length, 2*max_length, 3*max_length] given other factoids are negatives
            labels = torch.arange(bsz).to(device) * num_factoids
        else:
            # Does not use batch factoids for evaluation
            scores = self.score_factoids(query_embed=context_embed, factoid_embed=factoid_embed)
            labels = torch.zeros(bsz, dtype=torch.long).to(device)

        loss = nn.CrossEntropyLoss()(scores, labels)
        outputs = JointModelOutput(loss=loss, logits=scores.detach())

        return outputs


# Factory method to obtain relevant model
def get_model(
        model: AnyStr,
        mname: AnyStr
) -> nn.Module:
    models = {
        "memnet": MemNet,
        "supervised": SupervisedLearning,
        "polymemnet": PolyMemNet,
        "polyencoder": Polyencoder,
        "biencoder": BiEncoder,
        "rar": RetrievalAugmentedRetrieval
    }
    if model not in models:
        raise ValueError(f"Specified model: {model} not found; must select one of {list(models.keys())}.")

    return models[model].from_pretrained(mname)
