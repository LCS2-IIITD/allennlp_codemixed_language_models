{

 "train_data_path":
     std.toString({
        "lang1": "data/en_es/en_small.txt",
        "lang2": "data/en_es/es_small.txt",
        "cm": "data/en_es/train_mixed_big_small.txt",
    }),
f
 "validation_data_path":
     std.toString({
        "lang1": "data/en_es/test.txt",
        "lang2": "data/en_es/test.txt",
        "cm": "data/en_es/test.txt",
    }),

  "test_data_path":
     std.toString({
        "lang1": "data/en_es/test.txt",
        "lang2": "data/en_es/test.txt",
        "cm": "data/en_es/test.txt",
    }),

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

    "evaluate_on_test": true,
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },


    "model": {
        "type": "girnet_lm",
        "bidirectional": false,
        "contextualizer": {
//            "type": "bidirectional_language_model_transformer",
//            "dropout": 0.3,
//            "hidden_dim": 512,
//            "input_dim": 1024,
//            "input_dropout": 0.3,
//            "num_layers": 1
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
        "num_samples": 8126,
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
        "learning_rate_scheduler": {
            "type": "noam",
            "model_size": 512,
            "warmup_steps": 6000
        },
        "num_epochs": 10,
        "optimizer": {
            "type": "dense_sparse_adam"
        },
        "should_log_learning_rate": true
    },
//    "vocabulary": {
//        "directory_path": "store/all_vocab/vocabulary/"
//    }
}