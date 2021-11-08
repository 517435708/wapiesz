import ai

# text
ai.datasets.loader.import_memes(refresh=False)
print(ai.datasets.loader.get_memes_dataframe())
# memes