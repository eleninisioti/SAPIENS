env_name = "wordcraft-multistep-no-goal-v0"


recipe_book_info = {"single path": {"path": "recipe_books/1path.json",
                                    "best_paths": ["a_8"],
                                    "npaths": 1,
                                    "best_reward": 36,
                                    "max_steps": 1000000,
                                    "early_step": 500000},

                    "merging paths": {"path": "recipe_books/cross_easier.json",
                                      "npaths": 3,
                                      "best_reward": 50,
                                      "max_steps": 1000000,
                                      "early_step": 500000,
                                      "best_paths": ["a_8", "b_8", "c_10"]},

                    "best_of_ten": {"path": "recipe_books/10_mirror.json",
                                    "npaths": 10,
                                    "best_reward": 72,
                                    "max_steps": 3500000,
                                    "early_step": 500000
                                    },
                    }

