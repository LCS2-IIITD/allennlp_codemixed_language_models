{
  "train_data_path": "data/sentiment_analysis/train.txt",
  "test_data_path": "data/sentiment_analysis/test.txt",
  "evaluate_on_test": true,
  "dataset_reader": {
    "type": "cm_sentiment_analysis",
     "token_indexers": {
              "elmo": {
                "type": "elmo_characters"
              }
      },
    "tokenizer": {
        "type": "word",
        "word_splitter": {
            "type": "just_spaces"
        }
    }
  },
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "elmo": {
          "type": "girnet_lm_token_embedder",
          "archive_file": "./store/en_es_perfect/",
          "dropout": 0.2,
          "bos_eos_tokens": ["<S>", "</S>"],
          "remove_bos_eos": true,
          "requires_grad": false
        }
      }
    },
    "seq2vec_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 2048,
      "hidden_size": 1024,
      "num_layers": 2
    },
    "dropout": 0.2
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": 16
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
    }
  }
}