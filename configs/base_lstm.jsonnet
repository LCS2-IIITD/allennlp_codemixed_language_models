local NUM_GPUS = 1;
{
  "train_data_path": "data/cm/all_train_50k.tx",
  "validation_data_path": "data/cm/cm_valid.txt",
  "test_data_path": "data/cm/cm_test.txt",
  "evaluate_on_test": true,

  "dataset_reader": {
        "type": "simple_language_modeling",
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
        "max_sequence_length": 32,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
   },

   "validation_dataset_reader": {
       "type": "simple_language_modeling",
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
        "max_sequence_length": 500,
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
    },

  "vocabulary": {
       "tokens_to_add": {
           "tokens": ["<S>", "</S>"],
           "token_characters": ["<>/S"]
       },
       "min_count": {"tokens": 3}
   },

  "model": {
    "type": "language_model",
    "bidirectional": false,
    "num_samples": 4096,
    "sparse_embeddings": true,
    "text_field_embedder": {
      // Note: This is because we only use the token_characters during embedding, not the tokens themselves.
      "allow_unmatched_keys": true,
      "token_embedders": {
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "num_embeddings": 262,
                // Same as the Transformer ELMo in Calypso. Matt reports that
                // this matches the original LSTM ELMo as well.
                "embedding_dim": 16
            },
            "encoder": {
                "type": "cnn-highway",
                "activation": "relu",
                "embedding_dim": 16,
                "filters": [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 1024]],
                "num_highway": 2,
                "projection_dim": 512,
                "projection_location": "after_highway",
                "do_layer_norm": true
            }
        }
      }
    },
    // TODO(brendanr): Consider the following.
    // remove_bos_eos: true,
    // Applies to the contextualized embeddings.
    "dropout": 0.1,
    "contextualizer": {
            "type": "lstm",
            "bidirectional": false,
            "dropout": 0.33,
            "hidden_size": 512,
            "input_size": 512,
            "num_layers": 1
    }
  },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
  "trainer": {
   "validation_metric": "-perplexity",
    "num_epochs": 15,
    "patience":2,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "dense_sparse_adam"
    },
     "grad_norm": 10.0,
    "should_log_learning_rate": true
  }
}
