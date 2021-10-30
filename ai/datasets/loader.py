import pandas as pd
import os


# TODO:
#   - import data

def import_cornell():
    movie_lines_features = ["LineID", "Character", "Movie", "Name", "Line"]

    dir = os.path.dirname(__file__)
    base_dir = '../../datasets/cornell movie-dialogs corpus/'

    movie_lines = pd.read_csv(
        os.path.join(dir, base_dir + 'movie_lines.txt'),
        encoding='iso-8859-1',
        sep="\+\+\+\$\+\+\+",
        engine="python",
        index_col=False, names=movie_lines_features)

    movie_lines = movie_lines[["LineID", "Line"]]

    # Strip the space from "LineID" for further usage and change the datatype of "Line"
    movie_lines["LineID"] = movie_lines["LineID"].apply(str.strip)
    print(movie_lines.head())

    movie_conversations_features = ["Character1", "Character2", "Movie", "Conversation"]
    movie_conversations = pd.read_csv(os.path.join(dir, base_dir + 'movie_conversations.txt'),
                                      encoding='iso-8859-1',
                                      sep="\+\+\+\$\+\+\+",
                                      engine="python",
                                      index_col=False, names=movie_conversations_features)

    # Again using the required feature, "Conversation"
    movie_conversations = movie_conversations["Conversation"]
    print(movie_conversations.head())
