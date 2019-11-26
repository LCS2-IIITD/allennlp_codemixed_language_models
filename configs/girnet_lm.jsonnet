{
    "train_data_path":
     std.toString({
        "lang1": "data/en_es/en.txt",
        "lang2": "data/en_es/es.txt",
        "cm": "data/en_es/train.txt",
    }),


  "dataset_reader": {
            "type": "girnet_lm_reader",
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

  "vocabulary": {
    "min_count": {"tokens": 10}
  },
}