"""
Constructs the PersonaChat and Wizard-of-Wikipedia datasets.  Both files are downloaded from ParlAI: https://parl.ai/

See the papers for the respective datasets for more info:

Persona Chat: https://arxiv.org/pdf/2004.05388.pdf
Wizard of Wikipedia: https://arxiv.org/pdf/1811.01241.pdf
"""

import os
import tarfile
import jsonlines
import re
import json
import abc
import numpy as np
import copy
import requests

from itertools import chain
from typing import AnyStr
from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SEP_TOKEN = "[SEP]"


class Preprocessor(abc.ABC):

    @property
    @abc.abstractmethod
    def file_name(self):
        pass

    def __init__(self, tokenizer, dry_run=False, force_overwrite=False, load_from_disk=None):
        self.tokenizer = tokenizer
        self.max_factoids = None  # Maximum number of factoids contained in dataset
        self.dry_run = dry_run  # Whether to save data
        self.force_overwrite = force_overwrite

        if load_from_disk is None:
            # Create directory to store training/valid/test data
            if not os.path.isdir("src/data"):
                os.mkdir("src/data")

            # Load data if not already found
            for split in ["train", "valid", "test"]:
                if not os.path.isfile(f"src/data/{self.file_name}.{split}.jsonl") or self.dry_run or self.force_overwrite:
                    self.load_data(split)

        self.load_from_disk = load_from_disk

    @staticmethod
    def download_tar(url, dest):
        if not os.path.isdir("Datasets"):
            os.mkdir("Datasets")

        os.mkdir(os.path.join("Datasets", dest))
        print(f"Attempting to download dataset: {url}")
        request = requests.get(url, stream=True)
        if request.status_code != requests.codes.ok:
            raise ConnectionError(f"Could not connect: status code {request.status_code}")

        save_path = os.path.join("Datasets", dest, ".tar.gz")
        with open(save_path, "wb") as fd:
            for chunk in request.iter_content(chunk_size=128):
                fd.write(chunk)

        # Un-tar file
        new_path = os.path.join("Datasets", dest)
        tar = tarfile.open(save_path)
        tar.extractall(new_path)
        tar.close()

        print(f"Successfully downloaded and un-tarred: {url}")

    def tokenize(self, texts, max_length=32):
        return self.tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="np")

    def get_max_factoids(self, file):
        """Returns maximum number of factoids contained in the factoids column for a given dataset."""
        lengths = []
        for sample in jsonlines.open(file):
            lengths.append(len(sample["factoids"].split(SEP_TOKEN)))
        max_length = max(lengths)

        return max_length

    def map_fn(self, samples, split):
        bsz = len(samples["context"])
        #num_facts = 8 if split == "train" else self.max_factoids
        num_facts = self.max_factoids

        # Pad factoids up to max_factoids
        tmp_factoids = []
        lengths = []
        for sublist in samples["factoids"]:
            if split == "train":
                facts = sublist.split(SEP_TOKEN)[:num_facts]
                lengths.append(len(facts))
                facts += [""] * (num_facts - len(facts))
            else:
                facts = sublist.split(SEP_TOKEN)
                lengths.append(len(facts))
                facts += [""] * (self.max_factoids-len(facts))
            tmp_factoids.append(facts)
        factoids = [item for sublist in tmp_factoids for item in sublist]

        context_tokens = self.tokenize(samples["context"], max_length=128)
        response_tokens = self.tokenize(samples["response"])
        factoid_tokens = self.tokenize(factoids)

        factoid_tokens.input_ids = np.reshape(factoid_tokens.input_ids, (bsz, num_facts, -1))
        factoid_tokens.attention_mask = np.reshape(factoid_tokens.attention_mask, (bsz, num_facts, -1))

        outputs = {
            "x_input_ids": context_tokens.input_ids,
            "x_attention_mask": context_tokens.attention_mask,
            "z_input_ids": factoid_tokens.input_ids,
            "z_attention_mask": factoid_tokens.attention_mask,
            "y_input_ids": response_tokens.input_ids,
            "y_attention_mask": response_tokens.attention_mask,
            "lengths": lengths,
        }

        if "candidates" in samples and samples["candidates"][0] is not None:
            num_candidates = len(samples["candidates"][0].split(SEP_TOKEN))
            candidates = [item for sublist in samples["candidates"] for item in sublist.split(SEP_TOKEN)]
            candidate_tokens = self.tokenize(candidates)

            candidate_tokens.input_ids = np.reshape(candidate_tokens.input_ids, (bsz, num_candidates, -1))
            candidate_tokens.attention_mask = np.reshape(candidate_tokens.attention_mask, (bsz, num_candidates, -1))
            outputs["candidate_input_ids"] = candidate_tokens.input_ids
            outputs["candidate_attention_mask"] = candidate_tokens.attention_mask

        return outputs

    @staticmethod
    @abc.abstractmethod
    def load_data(split: AnyStr = "train"):
        pass

    def get_dataset(self):
        # Obtains Dataset object from jsonl file
        fpath = self.file_name + ".dataset_dict"
        if self.load_from_disk is not None:
            fpath = os.path.join(self.load_from_disk, fpath)

        root = os.path.join("src", "data")
        ftype = ".jsonl"
        splits = ("train", "valid", "test")

        # Get str path for train, valid, test
        train_path, valid_path, test_path = map(lambda x: os.path.join(root, self.file_name+"."+x+ftype), splits)
        paths = (train_path, valid_path, test_path)
        if not os.path.exists(fpath) or self.dry_run or self.force_overwrite:
            train, valid, test = map(lambda x: Dataset.from_json(x), paths)  # Get Dataset objects
            self.max_factoids = max(map(lambda x: self.get_max_factoids(x), paths))
            train = train.map(lambda x: self.map_fn(x, "train"), batched=True, batch_size=100,
                              load_from_cache_file=False)
            valid = valid.map(lambda x: self.map_fn(x, "valid"), batched=True, batch_size=100,
                              load_from_cache_file=False)
            test = test.map(lambda x: self.map_fn(x, "test"), batched=True, batch_size=100,
                              load_from_cache_file=False)

            dataset_dict = DatasetDict({"train": train, "valid": valid, "test": test})

            if not self.dry_run:
                dataset_dict.save_to_disk(fpath)
            else:
                exit("Dry run complete")  # dry run is complete
        else:
            dataset_dict = DatasetDict.load_from_disk(fpath)

        return dataset_dict


class PersonaChatPreprocessor(Preprocessor):

    file_name = "pc.original"
    persona = "original"

    def get_contexts_persona_responses(self, f, split):
        persona_a = []
        personae_a = []
        persona_b = []
        personae_b = []

        dialog = []
        contexts = []
        responses = []
        persona = []
        candidates = []

        reading_persona = True
        lines = f.readlines()
        for line in lines:
            if "your persona:" in line:
                if reading_persona is False:
                    personae_a.append(persona_a)
                    personae_b.append(persona_b)
                    persona_a = []
                    persona_b = []
                    dialog = []
                    reading_persona = True
                persona_a.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line).replace("\n", ""))
            elif "partner's persona:" in line:
                persona_b.append(re.sub(r"\A[0-9]+ (your persona: |partner's persona: )", "", line))
            else:
                # utterance line is split into speaker A + \t + speaker B + \t\t + candidate_1|candidate_2 etc.
                utts = line.split("\t")
                c = utts[3].replace("\n", "").split("|") if split != "train" else None  # No candidates during training
                context = re.sub(r"\A[0-9]+ ", "", utts[0])  # remove line numbering
                response = re.sub(r"\A[0-9]+ ", "", utts[1])
                dialog.append(context)
                contexts.append(SEP_TOKEN.join(dialog[-min(4, len(dialog)):]))
                responses.append(response)
                persona.append(persona_a)
                c.remove(response)  # Remove ground truth
                candidates.append(c)
                reading_persona = False

        return contexts, persona, responses, candidates

    def load_data(self, split: AnyStr = "train"):
        """
            Load dataset from Persona Chat paper: https://arxiv.org/abs/1801.07243
            :return: list of contexts, list of responses, list of personae
            """
        if not os.path.exists("Datasets/persona_chat"):
            self.download_tar('http://parl.ai/downloads/convai2/convai2_fix_723.tgz', "persona_chat")

        split2file = {
            "train": ("train_both_revised.txt", "train_both_original.txt"),
            "valid": ("valid_both_revised.txt", "valid_both_original.txt"),
            "test": ("valid_both_revised.txt", "valid_both_original.txt")  # Test data is not released publically
        }

        if self.persona == "original":
            split_idx = 1
        else:
            split_idx = 0

        with open(f"Datasets/persona_chat/{split2file[split][split_idx]}") as f:
            contexts, persona, responses, candidates = self.get_contexts_persona_responses(f, split)

        # Select pseudo-labels
        # train TF-IDF model
        if split == "train":
            tfidf = TfidfVectorizer()
            tfidf.fit(contexts+responses)
            tmp_personas = []
            for p, r in zip(persona, responses):
                query = tfidf.transform([r])
                keys = tfidf.transform(p)
                scores = linear_kernel(query, keys)
                label = p[np.argmax(scores, axis=-1)[0]]
                tmp_persona = copy.copy(p)
                tmp_persona.remove(label)
                tmp_persona.insert(0, label)
                tmp_personas.append(tmp_persona)
            persona = tmp_personas

        if not self.dry_run:
            if split == "train":
                jsonl = [{"context": c, "response": r, "factoids":  SEP_TOKEN.join(p)} for c, r, p in zip(contexts, responses, persona)]
            else:
                jsonl = [{"context": c,
                          "response": r,
                          "factoids":  SEP_TOKEN.join(p),
                          "candidates": SEP_TOKEN.join(cand)} for c, r, p, cand in zip(contexts, responses, persona, candidates)]
            with jsonlines.open("src/data/"+self.file_name+"."+split+".jsonl", mode="w") as writer:
                writer.write_all(jsonl)


class PersonaChatRevisedPreprocessor(PersonaChatPreprocessor):

    file_name = "pc.revised"
    persona = "revised"


class WizardofWikipediaPreprocessor(Preprocessor):

    tfidf_retrieval = "context"
    file_name = "wow"

    def load_data(self, split: AnyStr = "train"):
        if not os.path.exists("Datasets/wizard_of_wikipedia"):
            self.download_tar("http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz", "wizard_of_wikipedia")

        split2file = {
            "train": "train.json",
            "valid": "valid_random_split.json",
            "test": "test_random_split.json",
            "test_unseen": "test_topic_split.json"
        }

        EPISODE_END_TOKEN = "[END]"

        data = json.load(open(f"Datasets/wizard_of_wikipedia/{split2file[split]}"))
        contexts = []
        responses = []
        golden = []
        facts = []
        candidates = []
        for dialog in data:
            d = dialog["dialog"]
            for i in range(len(d)-1):
                if d[i]["speaker"] == "1_Apprentice" or d[i]["speaker"] == "0_Apprentice":
                    if i != 0:
                        contexts.append(d[i]["text"])
                        responses.append(d[i+1]["text"])

                        # Add negative candidates for evaluation
                        if split != "train":
                            c = d[i+1]["candidate_responses"]
                            c.remove(d[i+1]["text"])  # Remove ground truth
                            candidates.append(c)
                        else:
                            candidates.append(None)

                        try:
                            g = list(d[i+1]["checked_sentence"].values())[0]
                            # Sometimes no_passages_used is given as text, sometimes field is left blank
                            if g == "no_passages_used":
                                g = None
                        except IndexError:
                            g = None
                        golden.append(g)
                        sub_facts = copy.copy(dialog["chosen_topic_passage"])

                        if i > 0:
                            iter_passages = chain(d[i-1]["retrieved_passages"], d[i]["retrieved_passages"])
                        else:
                            iter_passages = d[i]["retrieved_passages"]

                        for topic in iter_passages:
                            for f in topic.values():
                                sub_facts += copy.copy(f)
                        sub_facts = list(dict.fromkeys(sub_facts))
                        if g is not None and g not in sub_facts:
                            print(g)
                            sub_facts += g
                        facts.append(sub_facts)
                elif i == 0 and d[0]["speaker"] == "0_Wizard":   # For start of conversation just use chosen topic as prompt
                    contexts.append(dialog["chosen_topic"])
                    responses.append(d[0]["text"])

                    # Add negative candidates for evaluation
                    if split != "train":
                        c = d[i + 1]["candidate_responses"]
                        c.remove(d[i + 1]["text"])  # Remove ground truth
                        candidates.append(c)
                    else:
                        candidates.append(None)

                    try:
                        g = list(d[0]["checked_sentence"].values())[0]
                        # Sometimes no_passages_used is given as text, sometimes field is left blank
                        if g == "no_passages_used":
                            g = None
                    except IndexError:
                        g = None
                    golden.append(g)
                    sub_facts = copy.copy(dialog["chosen_topic_passage"])
                    if g is not None and g not in sub_facts:
                        print(g)
                        sub_facts += g
                    facts.append(sub_facts)

            # Marks end of dialogue
            contexts.append(EPISODE_END_TOKEN)
            responses.append(EPISODE_END_TOKEN)
            golden.append(EPISODE_END_TOKEN)
            facts.append(EPISODE_END_TOKEN)
            candidates.append(EPISODE_END_TOKEN)

        # Construct multi-turn contexts
        tmp_contexts = []  # Stores all mt contexts
        tmp_responses = []
        tmp_golden = []
        tmp_facts = []
        tmp_candidates = []
        dialog = []  # For specific episode
        max_turns = 2
        for c, r, g, f, C in zip(contexts, responses, golden, facts, candidates):
            if c != EPISODE_END_TOKEN:
                dialog.append(c)  # Apprentice turn
                tmp_contexts.append(SEP_TOKEN.join(dialog[-min(max_turns, len(dialog)):]))
                dialog.append(r)  # Wizard turn
                tmp_responses.append(r)
                tmp_golden.append(g)
                tmp_facts.append(f)
                tmp_candidates.append(C)
            else:
                dialog = []
        contexts = tmp_contexts
        responses = tmp_responses
        golden = tmp_golden
        facts = tmp_facts
        candidates = tmp_candidates

        # For Training data, select pseudo-labels, otherwise use real labels for valid and test
        if split == "train":
            # Select pseudo-labels for knowledge
            # train TF-IDF model
            tfidf = TfidfVectorizer()
            tfidf.fit(contexts + responses)
            golden = []
            queries = contexts if self.tfidf_retrieval == "context" else responses
            for p, c in zip(facts, queries):
                if len(p) == 0:
                    golden.append(None)
                    continue
                query = tfidf.transform([c])
                keys = tfidf.transform(p)
                scores = linear_kernel(query, keys)
                label = p[np.argmax(scores, axis=-1)[0]]
                golden.append(label)

        # Remove label from negatives
        tmp_facts = []
        for g, sub_facts in zip(golden, facts):
            if g is not None:
                try:
                    sub_facts.remove(g)
                except:
                    pass
                sub_facts.insert(0, g)
            tmp_facts.append(sub_facts)
        facts = tmp_facts

        if not self.dry_run:
            jsonl = [
                {
                    "context": c,
                    "response": r,
                    "factoids": SEP_TOKEN.join(p),
                    "candidates": SEP_TOKEN.join(C) if C is not None else C
                } for c, r, p, g, C in zip(contexts, responses, facts, golden, candidates)
                if len(p) > 1 and golden is not None]

            with jsonlines.open("src/data/"+self.file_name+"."+split+".jsonl", mode="w") as writer:
                writer.write_all(jsonl)


class WizardofWikipediaOraclePreprocessor(WizardofWikipediaPreprocessor):

    tfidf_retrieval = "response"
    file_name = "wow.oracle"


def create_wow_unseen_dataset(tokenizer):
    """Run this to save the test unseen WoW data to disk.  Run with main project folder as WD."""
    processor = WizardofWikipediaOraclePreprocessor(tokenizer)
    processor.load_data("test_unseen")
    dataset = Dataset.from_json("src/data/wow.oracle.test_unseen.jsonl")
    processor.max_factoids = processor.get_max_factoids("src/data/wow.oracle.test_unseen.jsonl")
    dataset = dataset.map(lambda x: processor.map_fn(x, "test"), batched=True, batch_size=100,
                          load_from_cache_file=False)
    dataset.save_to_disk("wow.oracle.test_unseen.dataset")


# Factory method to obtain relevant dataset
def get_dataset(dataset_name: AnyStr, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    dataset_names = {
        "pc.original": PersonaChatPreprocessor,
        "pc.revised": PersonaChatRevisedPreprocessor,
        "wow": WizardofWikipediaOraclePreprocessor,
    }
    if dataset_name not in dataset_names:
        raise ValueError(f"Specified dataset: {dataset_name} not found; must select one of {list(dataset_names.keys())}.")

    dataset_dict = dataset_names[dataset_name](tokenizer).get_dataset()

    return dataset_dict
