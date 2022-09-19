from conspiracies.supernodes import stem


def test_stemming():
    test_phrases = "he goes to school, beautiful weather, democratic society, democrats"
    stemms = stem(test_phrases)

    assert isinstance(stemms, str)
    assert stemms == "he go to school , beauti weather , democrat societi , democrat"
