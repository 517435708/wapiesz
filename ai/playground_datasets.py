import ai

# memes
ai.datasets.loader.import_memes(refresh=True)
print(ai.datasets.loader.get_memes_dataframe())
