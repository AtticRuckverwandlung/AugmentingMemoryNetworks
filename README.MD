# Augmenting Memory Networks for Rich and Efficient Retrieval in Grounded Dialogue

This is the code for the anonymised paper: Augmenting Memory Networks for Rich and Efficient Retrieval in Grounded Dialogue.

## Requirements

The code has the following requirements:

* transformers==4.6.0
* numpy==1.18.5
* datasets==1.11.0
* torch==1.8.0
* jsonlines==2.0.0
* scikit-learn==0.23.2

To install them, run the command:

    $ pip install -r requirements.txt


## Training

The models can be trained by running the train.py file.

    python train.py --output_dir path/to/model \
                    --do_eval True \
                    --evaluation_strategy epoch \
                    --save_strategy epoch \
                    --dataset pc.original \
                    --mname memnet


## Testing

We can similarly evaluate our finetuned models by running the test.py file.

    python test.py --test_dataset_path path/to/test/dataset \
                   --pretrained_model path/to/model \
                   --mname memnet
