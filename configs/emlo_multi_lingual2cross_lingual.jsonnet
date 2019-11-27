{
    "train_data_path": "data/en_es/train_small.txt",
    "validation_data_path": "data/en_es/valid.txt",
    "test_data_path": "data/en_es/test.txt",

    "dataset_reader": {
        "type": "elmo_multi2cross_lm_reader",
        "max_sequence_length": 32,
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
              }
        },
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        },
        "start_tokens": ["<S>"],
        "end_tokens": ["</S>"]
    },

    "iterator": {
        "type": "basic",
        "batch_size": 32,
    },

    "model": {
        "type": "elmo_multi2cross_lm",
        "languages": ["en", "es"],
        "dropout": 0.33,
        "bidirectional": false,
        "contextualizer": {
            "type": "lstm",
            "bidirectional": false,
            "dropout": 0.33,
            "hidden_size": 2048,
            "input_size": 2048,
            "num_layers": 3
        },
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder_multilang",
                    "aligning_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_best_mapping.pth",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_best_mapping.pth",
//                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_best_mapping.pth",
//                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_best_mapping.pth",
//                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_best_mapping.pth",
//                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_best_mapping.pth",
//                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_best_mapping.pth"
                    },
                    "do_layer_norm": false,
                    "dropout": 0.3,
                    "scalar_mix_parameters": [
                        -9e10,
                        1,
                        -9e10
                    ],
                    "options_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
//                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
//                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
//                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
//                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json",
//                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/options262.json"
                    },
                    "weight_files": {
                        "en": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/en_weights.hdf5",
                        "es": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/es_weights.hdf5",
//                        "fr": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/fr_weights.hdf5",
//                        "it": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/it_weights.hdf5",
//                        "pt": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/pt_weights.hdf5",
//                        "sv": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/sv_weights.hdf5",
//                        "de": "https://s3-us-west-2.amazonaws.com/allennlp/models/multilingual_elmo/de_weights.hdf5"
                    }
                }
            }
        }
    },

   "trainer": {
        "num_epochs": 20,
        "cuda_device": -1,
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