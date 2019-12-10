local NUM_GPUS = 1;
{
  "train_data_path": "data/cm/all_train.txt",
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
       "min_count": {"tokens": 2}
   },

  "model": {
    "type": "language_model",
    "bidirectional": true,
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
        "type": "bidirectional_language_model_transformer",
        "input_dim": 512,
        "hidden_dim": 1024,
        "num_layers": 2,
        "dropout": 0.1,
        "input_dropout": 0.1
//            "type": "lstm",
//            "bidirectional": true,
//            "dropout": 0.33,
//            "hidden_size": 1024,
//            "input_size": 512,
//            "num_layers": 1
    }
  },
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
  "trainer": {
    "num_epochs": 9,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      // The gradient accumulators in Adam for the running stdev and mean for
      // words not used in the sampled softmax would be decayed to zero with the
      // standard "adam" optimizer.
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
