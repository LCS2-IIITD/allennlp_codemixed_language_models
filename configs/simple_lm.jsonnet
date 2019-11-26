{
    "train_data_path": "data/en_es/train_small.txt",
    "validation_data_path": "data/en_es/valid.txt",
    "test_data_path": "data/en_es/test.txt",
  "dataset_reader": {
    "type": "simple_lm_reader",
    "max_sequence_length": 32,
    "tokenizer": {
        "type": "word",
        "word_splitter": {
            "type": "just_spaces"
          }
    },
           "token_indexers": {
          "tokens": {
            "type": "single_id"
          },
          "token_characters": {
            "type": "elmo_characters"
          }
        },
    "start_tokens": ["<S>"],
    "end_tokens": ["</S>"]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "model": {
    "type": "simple_lm",
    "bidirectional": true,
    "num_samples": 8124,
    "sparse_embeddings": true,
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "num_embeddings": 262,
            "embedding_dim": 16
          },
          "encoder": {
            "type": "cnn-highway",
            "activation": "relu",
            "embedding_dim": 16,
            "filters": [
              [
                1,
                32
              ],
              [
                2,
                32
              ],
              [
                3,
                64
              ],
              [
                4,
                128
              ],
              [
                5,
                256
              ],
              [
                6,
                512
              ],
              [
                7,
                1024
              ]
            ],
            "num_highway": 2,
            "projection_dim": 256,
            "projection_location": "after_highway",
            "do_layer_norm": true
          }
        }
      }
    },
    // TODO(brendanr): Consider the following.
    // remove_bos_eos: true,
    // Applies to the contextualized embeddings.
    "dropout": 0.3,
    "contextualizer": {
      "type": "bidirectional_language_model_transformer",
      "input_dim": 256,
      "hidden_dim": 1024,
      "num_layers": 3,
      "dropout": 0.3,
      "input_dropout": 0.3
    }
  },
  "trainer": {
    "num_epochs": 20,
    "cuda_device": 0,
    "optimizer": {
      "type": "dense_sparse_adam"
    },
    // TODO(brendanr): Needed with transformer too?
    // "grad_norm": 10.0,
    "learning_rate_scheduler": {
      "type": "noam",
      // See https://github.com/allenai/calypso/blob/master/calypso/train.py#L401
      "model_size": 512,
      // See https://github.com/allenai/calypso/blob/master/bin/train_transformer_lm1b.py#L51.
      // Adjusted based on our sample size relative to Calypso's.
      "warmup_steps": 6000
    },
    "should_log_learning_rate": true
  }
}