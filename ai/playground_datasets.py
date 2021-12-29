import ai.datasets.loader as loader
"""
    Before any work set all True
"""
# memes dataset
reload = False
if reload:
    loader.import_memes()

upload_images = False
if upload_images:
    loader.download_images()

loader.prepare_text(with_plots=True)

print(loader.get_memes_dataframe())