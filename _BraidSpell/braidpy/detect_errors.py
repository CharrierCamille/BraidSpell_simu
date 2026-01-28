def subst(str1, str2, c1, c2):
    """
    It checks if the string str1 can be transformed into str2 by replacing one character c1 with another character c2

    :param str1: the first string
    :param str2: the string to be corrected
    :param c1: the character in str1 that is being replaced
    :param c2: the character that is in the correct position in the second string
    :return: True or False
    """
    idx = [i for i, x in enumerate(str1) if x == c1]
    for i in idx:
        if str1[:i] + c2 + (str1[i + 1:] if i < len(str1) - 1 else "") == str2:
            return True
    return False


def detect_substitution_n_error(stim, str1, str2):
    """
    Checks whether the substitution of a phoneme in the stimulus is a substitution where both phonemes usually share a letter (like a/an etc)
    """
    """ substitution in an "on", "an" etc grapheme
    """
    letters = {'a': 'a@', 'i': 'i5', 'o': 'o&', 'u': '5y', 'e': '@e'}
    for l, ph in letters.items():
        if l + 'n' in stim[:-1] or l + 'm' in stim[:-1]:  # il faut qu'il y ait encore une lettre après
            if subst(str1, str2, ph[0], ph[1]) or subst(str1, str2, ph[1], ph[0]):
                return True
    return False


def detect_substitution_schwa_error(str1, str2):
    """
    Checks if there is a schwa substitution
    """
    grapheme_list = ['i', '5', 'y', 'e', 'a', 'o', '@', '&', '2']
    for g in grapheme_list:
        if subst(str1, str2, g, '°') or subst(str1, str2, '°', g):
            return True
    return False


def detect_substitution_grapheme_error(str1, str2):
    """
    Checks if the two strings differ by a single phoneme, and if the difference is one of the grapheme errors

    """
    grapheme_list = ['i5', 'yu', 'ij', 'ie', 'ao', 'a@', 'ae', 'o&', 'ow', 'ks', 'sz', '°e', 'gZ', '2e', 'st']
    for g in grapheme_list:
        if subst(str1, str2, g[0], g[1]) or subst(str1, str2, g[1], g[0]):
            return True
    return False


def detect_insertion_grapheme_error(str1):
    """ Detects insertion error that corresponds to a grapheme error
    """
    grapheme_list = ['i5', 'ij', 'ie', 'ao', 'a@', 'ae', 'o&', 'ow', 'ks', 'sz', '°e', 'gZ', '2e']
    for g in grapheme_list:
        if g in str1 or g[::-1] in str1:
            return True
    return False


def detect_substitution_error(str1, str2):
    """ Detects substitution error
    """
    cpt = 0
    if len(str1) == len(str2):
        for i, j in zip(str1, str2):
            if i != j:
                cpt += 1
    return cpt == 1


def detect_insertion_error(str1, str2):
    """ Detects insertion error
    """
    for i in range(len(str2)):
        str2_tmp = str2[:i] + (str2[i + 1:] if i < len(str2) - 1 else "")
        if str2_tmp == str1:
            return i, True
    return 0, False

def detect_end_error(str1, str2):
    """
    Detects substitution error at the last position

    """
    return str2[:-1] == str1[:-1]

def detect_deletion_error(str1, str2):
    """ Detects deletion error
    """
    i,val = detect_insertion_error(str2, str1)
    if val:
        cons_phono = ['b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'R', 's', 't', 'v', 'z', 'S', 'G', 'J', 'N']
        if len(str1) == len(str2)+1 and str2 == str1[:-1]:
            return "end deletion error"
        elif (i>0 and str1[i-1] in cons_phono) or (i<len(str1)-1 and str1[i+1] in cons_phono):
            return "consonant cluster deletion error"
        else:
            return "deletion error"
        return ""


