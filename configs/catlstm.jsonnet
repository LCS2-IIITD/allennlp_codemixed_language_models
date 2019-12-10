{

 "train_data_path":
     std.toString({
        "lang1": "data/cm/mono_en_train.txt",
        "lang2": "data/cm/mono_es_train.txt",
        "cm": "data/cm/mono_all_train.txt",
    }),

 "validation_data_path":
     std.toString({
        "lang1": "data/cm/cm_valid.txt",
        "lang2": "data/cm/cm_valid.txt",
        "cm": "data/cm/cm_valid.txt",
    }),

  "test_data_path":
     std.toString({
        "lang1": "data/cm/cm_test.txt",
        "lang2": "data/cm/cm_test.txt",
        "cm": "data/cm/cm_test.txt",
    }),

   "evaluate_on_test": true,

   "dataset_reader": {
        "type": "girnet_lm_reader",
        "end_tokens": [
            "</S>"
        ],
        "max_sequence_length": 32,
        "start_tokens": [
            "<S>"
        ],
        "token_indexers": {
            "token_characters": {
                "type": "elmo_characters"
            },
            "tokens": {
                "type": "single_id"
            }
        },
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        }
    },

   "validation_dataset_reader": {
        "type": "girnet_lm_reader",
        "end_tokens": [
            "</S>"
        ],
        "max_sequence_length": 500,
        "start_tokens": [
            "<S>"
        ],
        "token_indexers": {
            "token_characters": {
                "type": "elmo_characters"
            },
            "tokens": {
                "type": "single_id"
            }
        },
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        }
    },

    "iterator": {
        "type": "basic",
        "batch_size": 32
    },


    "model": {
        "type": "girnet_lm",
        "bidirectional": false,
        "contextualizer": {
            "type": "lstm",
            "bidirectional": false,
            "dropout": 0.33,
            "hidden_size": 512,
            "input_size": 1024,
            "num_layers": 1
        },
        "dropout": 0.3,
        "aux_contextualizer": {
            "type": "lstm",
            "bidirectional": false,
            "dropout": 0.33,
            "hidden_size": 512,
            "input_size": 512,
            "num_layers": 1
        },
        "num_samples": 4096,
        "sparse_embeddings": true,
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "token_embedders": {
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 16,
                        "num_embeddings": 262
                    },
                    "encoder": {
                        "type": "cnn-highway",
                        "activation": "relu",
                        "do_layer_norm": true,
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
                        "projection_dim": 512,
                        "projection_location": "after_highway"
                    }
                }
            }
        }
    },

    "trainer": {
        "cuda_device": 0,
        "validation_metric": "-ppl_cm",
        "num_epochs": 4,
        "optimizer": {
            "type": "dense_sparse_adam"
        },
        "should_log_learning_rate": true
    },
}