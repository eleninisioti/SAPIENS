env_name = "wordcraft-multistep-no-goal-v0"


recipe_book_info = {"single_path": {"path": "lib/wordcraft/recipe_books/single_path.json",
                                    "best_paths": ["a_8"],
                                    "n_paths": 1,
                                    "best_reward": 36,
                                    "max_steps": 1000000,
                                    "early_step": 500000},

                    "merging_paths": {"path": "lib/wordcraft/recipe_books/merging_paths.json",
                                      "n_paths": 3,
                                      "best_reward": 50,
                                      "max_steps": 1000000,
                                      "early_step": 500000,
                                      "best_paths": ["a_8", "b_8", "c_10"]},

                    "bestoften_paths": {"path": "lib/wordcraft/recipe_books/bestoften_paths.json",
                                    "n_paths": 10,
                                    "best_reward": 72,
                                    "max_steps": 3500000,
                                    "early_step": 500000
                                    }
                    }

