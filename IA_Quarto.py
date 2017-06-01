from math import sqrt
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np

#####################################
##  Configurations préconstruites  ##
#####################################

vide = [-1, -1, -1, -1]


# On note p = 16 le nombre de cases, pièces, et stade maximaux
#
# On défini les types :
#   case : couple d'entiers entre 0 et 3
#   piece (ou donnee) : quadruplet (liste en pratique) d'entiers valant 0 ou 1 (ou -1 ssi vide)
#   plateau : matrice 4*4 de pièces
#   stock : liste de pièces
#   config = (plateau,stock,donnee) respectant les invariants suivants :
#       l'ensemble des pièces non vides du plateau, le stock et la donnee forment une partition de {0,1}^4


def config_initiale():
    """
    ---Entrée---
    
    ---Sortie--- O(p)
    Renvoie une configuration initialisée 
    """
    plateau = -np.ones((4, 4, 4))
    stock = [np.array([i, j, k, l]) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]

    donnee = np.array([0, 0, 0, 0])
    stock.remove(donnee)
    return plateau, stock, donnee


# On considérera toujours que la donnee initiale est [0,0,0,0]


#######################################################
##  Fonctions intermédiaires sur les configurations  ##
#######################################################

def alignements(plateau):
    """
    ---Entrée---
    plateau : plateau 
    ---Sortie--- O(p)
    Renvoie la liste des alignements de pièces sur le plateau
    Dans l'ordre
        ligne1 en haut, ligne2, ligne3 et ligne4
        colonne1 à gauche, colonne2, colonne3, colonne4
        diagonale1 de haut à gauche à en bas à doite, diagonale2
    """
    aligns = []

    for i in range(4):
        aligns.append(plateau[i])  # lignes

    for i in range(4):
        aligns.append([plateau[k][i] for k in range(4)])  # colonnes

    aligns.append([plateau[k][k] for k in range(4)])  # diagonale 1
    aligns.append([plateau[k][3 - k] for k in range(4)])  # diagonale 2
    return aligns


def alignements_relatif(case, plateau):
    """
    ---Entrée---
    case : coordonnées (i,j) d'une case
    ---Sortie--- O(p)
    Renvoie la liste des alignements dont fait partie la case
    """
    alignsrel = []
    i, j = case

    alignsrel.append(plateau[i])  # lignes

    alignsrel.append([plateau[k][j] for k in range(4)])  # colonnes

    if i == j:
        alignsrel.append([plateau[k][k] for k in range(4)])  # diagonale 1
    if i == 3 - j:
        alignsrel.append([plateau[k][3 - k] for k in range(4)])  # diagonale 2

    return alignsrel


def similitudes(ali):
    """
    ---Entrée---
    ali : alignement (liste de 4 pièces)
    ---Sortie--- O(p)
    Renvoie la liste des aspects, où un aspect est un couple (une liste) [k,N] avec
        k la valeur de la caractéristique (0 ou 1), ou -1 si aucune valeur commune
        N le nombre de pièce alignées ayant cette caractéristique, ou quelconque si k == -1
        ou alors aspect = [] si ali ne contient que des vides
    """
    commun = []
    for indcar in range(4):  # pour chaque caractéristique
        aspect = []

        for piece in ali:  # examen des pièces de l'alignement
            valcar = piece[indcar]  # valeur de la caracteristique de la piece examinée

            if valcar != -1:  # si piece non vide

                if aspect == []:  # s'il n'y a eu que des pièces vides avant
                    aspect = [valcar, 1]

                else:  # s'il y a eu une pièce non vide avant

                    if valcar != aspect[0]:  # si la valeur examinée est différente de celle enregistrée
                        aspect[0] = -1
                        break  # il n'y a pas de points communs sur cette caractéristique
                    else:
                        aspect[1] += 1

        commun.append(aspect)

    return (commun)


def est_complet(ali):
    """
    ---Entrée---
    ali : alignement (liste de 4 pièces)
    ---Sortie--- O(p)
    Renvoie True ssi les 4 pièces de l'alignement ont un caractéristique commune
    """
    commun = similitudes(ali)

    for aspect in commun:  # couple [k,N] valcar,nombre
        if aspect != [] and aspect[0] != -1 and aspect[1] == 4:  # si non vide,
            # ayant une valeur commune, et 4 pièces de même valeur (pour la caractéristique/aspect donnée)
            return (True)

    return (False)


def contraintes(plateau):
    """
    ---Entrée---
    plateau : plateau
    ---Sortie--- O(p^2)
    Renvoie la liste des contraintes [indali, indcar, valcar]
    """
    contrs = []
    aligns = alignements(plateau)

    for indali, ali in enumerate(aligns):
        commun = similitudes(ali)

        for indcar, aspect in enumerate(commun):
            if aspect != [] and aspect[0] != -1 and aspect[1] == 3:
                # si alignement non vide, non obstrué et contraint pour indcar
                contrs.append([indali, indcar, aspect[0]])

    return (contrs)


def est_contrainte(piece, contrs):
    """
    ---Entrée---
    piece : pièce
    contrs : liste des contraintes [indali, indcar, valcar]
    ---Sortie--- O(p)
    Renvoie le stock, moins les pièces contraintes (càd perdantes)
    """
    for cont in contrs:
        if piece[cont[1]] == cont[2]:  # si la piece est contrainte
            return (True)

    return (False)


#####################################################
##  Opérations principales sur les configurations  ##
#####################################################

def copie_config(config):
    """
    ---Entrée---
    config : configuration
    ---Sortie--- O(p)
    Crée une copie de config, sans alias
    """
    plateau, stock, donnee = config

    copie_donnee = donnee[:]
    copie_stock = [piece[:] for piece in stock]
    copie_plateau = [[piece[:] for piece in ligne] for ligne in plateau]

    return (copie_plateau, copie_stock, copie_donnee)


def jouer_coup(config, case, piece):
    """
    ---Entrée---
    config : configuration
    case : case du support du plateau de config
    piece : pièce du stock de config
    ---Sortie--- O(p)
    Renvoie la configuration après avoir placé donnee sur case, et donné piece
    """
    plateau, stock, donnee = copie_config(config)
    i, j = case

    plateau[i][j] = donnee
    donnee = piece
    if donnee != vide:
        stock.remove(donnee)

    return plateau, stock, donnee


def support(plateau):
    """
    ---Entrée---
    plateau : plateau
    ---Sortie--- O(p)
    Renvoie l'ensemble des cases vides du plateau
    """
    return [(i, j) for i in range(4) for j in range(4) if plateau[i][j] == vide]


def est_gagne(plateau):
    """
    ---Entrée---
    plateau : plateau
    ---Sortie--- O(p^2)
    Renvoie True ssi le jeu est gagné
    """
    aligns = alignements(plateau)

    for ali in aligns:  # pour chaque alignement
        if est_complet(ali):
            return True

    return False


def est_fini(plateau):
    """
    ---Entrée---
    plateau : plateau
    ---Sortie--- O(p^2)
    Renvoie s'il y a un gagnant (1), si le match est nul (0) ou la partie est indéterminée (-1)
    """
    stade = 15 - len(support(plateau)) + 1
    # car len(supp) = len(stock) + 1 (donnée pas dans le stock)

    if est_gagne(plateau):
        return 1
    elif stade == 16:
        return 0
    else:
        return -1


def stock_reduit(plateau, stock):
    """
    ---Entrée---
    plateau : plateau
    stock : stock
    ---Sortie--- O(p^2)
    Renvoie le stock, moins les pièces contraintes (càd perdantes)
    """
    contrs = contraintes(plateau)

    return ([piece for piece in stock if not est_contrainte(piece, contrs)])


###########################
##  Recherche d'extrema  ##
###########################

def max_cond(l, cond):
    """
    ---Entrée---
    l : liste non vide
    cond : fonction booléenne sur les éléments de l
    ---Sortie--- O((len(l))
    Renvoie le maximum des valeurs satisfaisant cond
        ou l[0] si aucune ne satisfait cond
    """
    m = l[0]  # initialisation

    for i in range(len(l)):
        if cond(l[i]) and (l[i] > m or not cond(m)):
            m = l[i]

    return (m)


def min_cond(l, cond):
    """
    ---Entrée---
    l : liste non vide
    cond : fonction booléenne sur les éléments de l
    ---Sortie--- O((len(l))
    Renvoie le minimum des valeurs satisfaisant cond
        ou l[0] si aucune ne satisfait cond
    """
    m = l[0]  # initialisation

    for i in range(len(l)):
        if cond(l[i]) and (l[i] < m or not cond(m)):
            m = l[i]

    return (m)


def liste_produit(variables, k):
    """
    ---Entrée---
    variables : liste [V1; ...; Vn] de listes Vi d'éléments
    k : indice tel que Vk est en cours de déconstruction
    ---Sortie--- O(len(V1)*...*len(Vn))
    Renvoie le produit cartésien V1x...xVn sous forme d'une seule liste
    """
    if k == len(variables) - 1:
        return ([[v] for v in variables[k]])
    else:
        return ([[v] + R for v in variables[k] for R in liste_produit(variables, k + 1)])


def extrema(variables, evaluation, maximiser, maximise, minimise):
    """
    ---Entrée---
    variables : liste [V1, ..., Vn] de listes Vi non vide d'éléments
    evaluation : fonction d'évaluation de V1x...xVn (dont les éléments
    sont sous forme de liste) dans un ensemble ordonné E
    etatmax : booléen
        *True : on veut maximiser
        *False : on veut minimiser
    maximise : fonction d'une liste vers sont maximum selon l'ordre sur E
        (la plupart du temps max lui-même)
    minimise : fonction d'une liste vers sont minimum selon l'ordre sur E
        (la plupart du temps min lui-même)
    ---Sortie--- O(len(V1)*...*len(Vn) * C(evaluation) + C(maximise) + C(minimise))
    Renvoie (extr,tuple_extrs) l'extrema de evaluation pour des n-uplets variant dans V1x...xVn
    et la liste tuple_extrs des n-uplets où il est atteint
    """

    tuples = liste_produit(variables, 0)  # on traite en fait les tuples sous forme de liste
    # pour pouvoir les concaténer
    valeurs = [evaluation(v) for v in tuples]

    if maximiser:
        m = maximise(valeurs)
    else:
        m = minimise(valeurs)

    ind_extrs = [i for i, v in enumerate(valeurs) if v == m]

    return m, [tuples[i] for i in ind_extrs]


####################
##  Heuristiques  ##
####################

### Pour Empirique et Antijoueur

def calcule_valeur(config, case):
    """
    ---Entrée---
    config : configuration
    case : case libre
    ---Sortie--- O(p^2)
    Fonction d'évaluation
    Revoie la valeur d'un coup : entier entre 0 et 15
    Utilisé par Empirique et Antijoueur
    """
    plateau, stock, donnee = config
    i, j = case

    plateau[i][j] = donnee
    stockred = stock_reduit(plateau, stock)
    plateau[i][j] = vide  # on évite de faire une copie en retirant la donnee placée

    return (15 - len(stockred))


### Pour Minmax

def valeur_piece(config, piece):
    """
    ---Entrée---
    config : configuration
    piece : pièce (du stock)
    ---Sortie--- O(p^2)
    Fonction d'évaluation
    Calcule la valeur de la piece en tant que possible donnee :
        entier entre -1 et 480 (majorant non atteint) en pratique jusqu'à 200 maximum
    Utlisé par Heuristique
    """
    plateau, stock, donnee = config

    supp = support(plateau)
    stockred = stock_reduit(plateau, stock + [piece])

    if piece not in stockred:  # si la pièce est contrainte
        return (-1)

    else:
        somme = 0
        for case in supp:  # On somme sur l'ensemble des cases potentielles
            i, j = case
            plateau[i][j] = piece
            for ali in alignements_relatif(case, plateau):
                for aspect in similitudes(ali):
                    if aspect[0] != -1:
                        # si la piece posée sur cette case induit une caractéristique (aspect) non contraite
                        # sur l'alignement ali
                        somme += aspect[1]

            plateau[i][j] = vide  # on évite de faire une copie en retirant la donnee placée

    return (somme)


def maximise_h(l):
    """
    ---Entrée---
    l : liste de triplets (gagne,valcase,valpiece) avec
        gagne : 1 (gagnant), 0 (indéterminé ou match nul), -1 (perdant)
        valcase : entier entre 0 et 15
        valpiece : entier entre -1 et 480 (majorant non atteint) en pratique jusqu'à 200 maximum
    ---Sortie--- O(len(l))
    Renvoie le maximum de l (pour l'odrdre lexicographique),
        dans le cas où la configuration correspondante est
        gagnée (t[0] == 1)
        ou il existe une pièce non contrainte (t[1] != 15 et celle-ci convient t[2] != -1)
    """
    return (max_cond(l, lambda t: t[0] == 1 or (t[1] != 15 and t[2] != -1)))


def minimise_h(l):
    """
    ---Entrée---
    l : liste de triplets (gagne,valcase,valpiece) avec
        gagne : 1 (gagnant), 0 (indéterminé ou match nul), -1 (perdant)
        valcase : entier entre 0 et 15
        valpiece : entier entre -1 et 480 (majorant non atteint) en pratique jusqu'à 200 maximum
    ---Sortie--- O(len(l))
    Renvoie le minimun de l, dans le cas où la donnee configuration (la pièce en particulier) correspondante
        n'est pas contrainte (t[2] != -1)
    """
    return (min_cond(l, lambda t: t[2] != -1))


valeur_nul = 0  # valeur donnée à un match nul


# caractérise la "prise de risque"
def heuristique(config):
    """
    ---Entrée---
    config : configuration
    etatmax : booléen
        *True : on veut une contrainte maximale
        *False : on veut une contrainte minimale
    ---Sortie--- O(p^4)
    Fonction dévaluation
    Renvoie (val, case, piece)
    """
    plateau, stock, donnee = config
    supp = support(plateau)
    fin = est_fini(plateau)

    ### Cas d'une configuration finale
    if fin != -1:
        if fin == 1:  # si gagné
            return ((-1, 0, 0))
        else:  # si match nul
            return ((0, valeur_nul, 10 * valeur_nul))

    ### Cas d'une configuration avant-finale
    elif stock == []:
        fin = est_fini(jouer_coup(config, supp[0], vide)[0])
        if fin == 1:  # si gagné
            return ((1, 0, 0))
        else:  # si match nul
            return ((0, 16 - valeur_nul, 160 - 10 * valeur_nul))

    else:
        ###Recherche d'extrema
        def evaluation(tuple_c_p):
            """O(p^2)"""
            case, piece = tuple_c_p
            copie = jouer_coup(config, case, piece)
            a = int(est_gagne(copie[0]))
            b = calcule_valeur(copie, case)
            c = valeur_piece(copie, piece)
            return a, b, c
            # on privilégie les placements gagnants, puis les contraintes,
            # puis le potentiel de la piece donnée

        val, _ = extrema([supp, stock], evaluation, True, maximise_h, minimise_h)

        return (val)


def alphabeta(config, depth: int, alpha: float, beta: float):
    if depth == 0:
        a, b, c = heuristique(config)
        return a + b / 100
    else:
        plateau, stock, donnee = config
        supp = support(plateau)

        u = -float('inf')
        for piece in stock:
            for case in supp:
                val = -alphabeta(jouer_coup(config, case, piece), depth - 1, -alpha, -beta)
                if val > u:
                    u = val
                    if u > alpha:
                        alpha = u
                        if alpha >= beta:
                            return u
        return u


def valeur_minmax(config, etatmax, alpha, beta, profondeur):
    """
    ---Entrée---
    config : configuration
    etatmax : booléen
        *True : on veut maximiser
        *False : on veut minimiser
    alpha : maximum des valeurs des fils, dont on en cherche le minimum
    beta : minimum des valeurs des fils, dont on en cherche le maximum
    profondeur : nombre maximal de couches qu'il reste encore à explorer
    ---Sortie--- O(p^(4 + 2*profondeur)) car c_p = p²*c_p et c_0 = p^4
    Fonction d'évaluation
    Renvoie la valeur d'une configuration selon la méthode MinMax
        de profondeur maximale profondeur
        avec pour évaluation aux feuilles (et à la profondeur donnée) heuristique
        la valeur étant un triplets (gagne, valcase, valpiece) avec
            gagne : 1 (gagnant), 0 (indéterminé ou match nul), -1 (perdant)
            valcase : entier entre 0 et 15
            valpiece : entier entre -1 et 480 (majorant non atteint) en pratique jusqu'à 200 maximum
    Utilisée par Minmax
    """
    plateau, stock, donnee = config
    supp = support(plateau)

    ### Si la partie est gagnée, nulle ou la profondeur est atteinte:
    if profondeur <= 0 or est_fini(plateau) != -1:
        return heuristique(config)

    else:
        ### Coupe alpha
        if etatmax:
            for case in supp:
                for piece in stock:
                    copie = jouer_coup(config, case, piece)
                    alpha = maximise_h([alpha, valeur_minmax(copie, not (etatmax), alpha, beta, profondeur - 1)])

                    if alpha >= beta:  # s'il ne minimise pas la valeur déjà enregistrée
                        return alpha  # on arrête la recherche

            return alpha  # si pas de alpha trouvé supérieur au alpha entré en paramètre
            # le coup ne sera pas joué

        ### Coupe beta
        else:
            for case in supp:
                for piece in stock:
                    copie = jouer_coup(config, case, piece)
                    beta = minimise_h([beta, valeur_minmax(copie, not (etatmax), alpha, beta, profondeur - 1)])

                    if alpha >= beta:  # s'il ne maximise pas la valeur déjà enregistrée
                        return (beta)  # on arrête la recherche

            return beta  # si pas de beta trouvé inférieur au beta entré en paramètre
            # le coup ne sera pas joué


def minimax_s(config, p):
    plateau, stock, donnee = config
    supp = support(plateau)

    ### Si la partie est gagnée, nulle ou la profondeur est atteinte:
    if p <= 0 or est_fini(plateau) != -1:
        a, b, c = heuristique(config)
        return a + b / 100
    else:
        return np.max([[-minimax_s(jouer_coup(config, c, piece), p - 1) for c in supp] for piece in stock])


### Pour Prevoyante

def calcule_note(config, longprev):
    """
    ---Entrée---
    config : configuration
    longprev : nombre de partie continuée pour chaque coup
    ---Sortie--- O(p^4 * longprev)
    Fonction d'évaluation
    Calcule la note de la configuration : flottant entre -1 (perdant) et 1 (gagnant)
    Utilisé par Prevoyante
    """
    plateau, stock, donnee = config

    stade = 15 - len(stock)
    note = 0  # initialisation

    for i in range(longprev):
        copie = copie_config(config)  # pour éviter de modifier la configuration en cours
        gagnant, _ = jeu(copie, empirique, empirique)

        if gagnant != 0:  # si pas match nul
            if stade % 2 == gagnant - 1:
                # si c'est l'adversaire du joueur courant, qui gagne
                note -= 1
            else:
                note += 1

    return (note / longprev)


#######################################################
##  Actions principales des IA sur une configuration ##
#######################################################

def case_gagnante(config):
    """
    ---Entrée---
    config : configuration
    ---Sortie--- O(p^3)
    Renvoie la case du plateau qui fait gagner le joueur avec donnee
        ou (-1, -1) s'il n'y en a pas
    """
    plateau, stock, donnee = config
    supp = support(plateau)

    # on essaie les positions qui peuvent nous faire gagner
    # en modifiant le quarto lui-même (mais en effaçant après)

    for case in supp:
        i, j = case
        plateau[i][j] = donnee  # on essaie ce placement

        if est_gagne(plateau):
            plateau[i][j] = vide
            return (case)

        plateau[i][j] = vide  # on elève la pièce

    return ((-1, -1))


def action_contrainte(config, etatmax):
    """
    ---Entrée---
    config : configuration
    etatmax : booléen
        *True : on veut une contrainte maximale (Empirique)
        *False : on veut une contrainte minimale (Antijoueur)
    ---Sortie--- O(p^3)
    Modifie la configuration (joue un coup) en agissant sur les contraintes
    """
    plateau, stock, donnee = config
    supp = support(plateau)  # supposé non vide

    ###Recherche d'extrema

    def evaluation(tuple_case):
        return (calcule_valeur(config, tuple_case[0]))

    def maximise(l):
        return (max_cond(l, lambda v: v != 15))  # 15 correspond à un stock réduit vide

    _, tuple_extrs = extrema([supp], evaluation, etatmax, maximise, min)
    # un élément de tuple_extrs est un tuple d'une valeur (une case)

    ###Traitement du résultat obtenu 

    i, j = tuple_extrs[random.randint(0, len(tuple_extrs) - 1)][0]  # une des cases dont la valeur est extremale
    plateau[i][j] = donnee  # on place donnee
    stockred = stock_reduit(plateau, stock)

    if stockred == []:  # si l'adversaire gagne dans tous les cas
        donnee = stock[0]  # donnee n'importe pas
    else:
        donnee = stockred[random.randint(0, len(stockred) - 1)]
        # donnee aléatoire parmi celles qui ne font pas gagner l'adversaire

    stock.remove(donnee)
    return (plateau, stock, donnee)


def decision_minmax(config, profondeur):
    """
    ---Entrée---
    config : configuration
    profondeur : nombre de couches maximal qu'on veut explorer
    ---Sortie--- O(p^(6 + 2*profondeur))
    Renvoie le coup (case,piece) qui maximise la valeur selon MinMax
    """
    plateau, stock, donnee = config
    supp = support(plateau)

    ###Recherche des maxima

    # def evaluation(t):
    #     case, piece = t
    #     return valeur_m(jouer_coup(config, case, piece), profondeur)
    #     # return valeur_minmax(jouer_coup(config, case, piece), False, (-1, 0, 0), (1, 16, 480), profondeur)
    #
    # _, tuple_maxs = extrema([supp, stock], evaluation, True, maximise_h, minimise_h)
    #
    # ###Traitement du résultat
    #
    # i = random.randint(0, len(tuple_maxs) - 1)
    #
    # return tuple_maxs[i][:2]

    if len(stock) == 0:
        print("Empty")

    m = -float("inf")
    max_case = -10000
    max_piece = -10000
    for case in supp:
        for piece in stock:
            x = alphabeta(jouer_coup(config, case, piece), profondeur, -float("inf"), float("inf"))
            # x = valeur_minmax(config, False, (-1, 0, 0), (1, 16, 480), profondeur)
            # x = x[0] + x[1] / 100
            if x > m:
                m = x
                max_case = case
                max_piece = piece

    return max_case, max_piece


def prevoyance(config, etatmax):
    """
    ---Entrée---
    config : configuration
    etatmax : booléen
        *True : on veut une contrainte maximale
        *False : on veut une contrainte minimale (pas utilisé ici)
    ---Sortie--- O(p^6 * longprev)
    Renvoie la configuration après un coup joué qui maximise la note
    selon la méthode de Monte Carlo de paramètre
        longprev (nombre de parties continuées après chaque coup)
        et d'heuristique de parcours Empirique
    """
    plateau, stock, donnee = config
    supp = support(plateau)  # supposé non vide

    ###Aperçu rapide de la situation actuelle
    copie = copie_config(config)
    note = calcule_note(copie, int(sqrt(longprev)))  # note de la configuration actuelle, sans trop de précision
    if note <= -0.9 or note == 0.5:  # si on est presque sûr de gagner, ou si c'est toujours match nul
        return (empirique(config))

    else:
        ###Recherche des extrema

        def evaluation(tuple_c_p):
            case, piece = tuple_c_p
            return (calcule_note(jouer_coup(config, case, piece), longprev))

        _, tuple_extrs = extrema([supp, stock], evaluation, etatmax, max, min)
        # un élément de tuple_extrs est un tuple d'une case et d'une pièce

        ###Traitement du résultat

        i = random.randint(0, len(tuple_extrs) - 1)
        return (jouer_coup(config, tuple_extrs[i][0], tuple_extrs[i][1]))


##########
##  IA  ##
##########

# On apèle IA une fonction qui à une configuration 1 renvoie une configuration 2 (càd joue une coup), tel que
#    (1) l'invarient sur la configuration est maintenu
#    (2) plateau2 ne diffère de plateau1 que par le présence de donnee1 sur une case vide de plateau1
#    (3) vide est renvoyé comme donnee ssi stock1 est vide

def aleatoire(config):
    """
    ---Entrée---
    config : configuration
    ---Sortie--- O(p)
    IA qui joue totalement aléatoirement
    (en respectant ce qui est correct evidemment)
    Renvoie une configuration après un coup
    """
    plateau, stock, donnee = config
    supp = support(plateau)  # supposé non vide

    case = supp[random.randint(0, len(supp) - 1)]  # case au hasard
    if stock == []:  # s'il n'y a plus de pièces à donner
        piece = vide
    else:
        piece = stock[random.randint(0, len(stock) - 1)]  # piece au hasard

    return (jouer_coup(config, case, piece))


def ia(config, etatmax):
    """
    ---Entrée---
    config : configuration
    etatmax : booléen
        *True : on veut une contrainte maximale (Empirique)
        *False : on veut une contrainte minimale (Antijoueur)
    ---Sortie--- O(p^3)
    IA qui maximise ou minimise la valeur (calcule_valeur) du coup
    Renvoie une configuration après un coup
    """
    plateau, stock, donnee = config

    if stock == []:  # s'il n'y a plus de pièces à donner
        return (jouer_coup(config, support(plateau)[0], vide))

    else:
        case = case_gagnante(config)
        if case != (-1, -1):  # si on gagne
            return (jouer_coup(config, case, stock[0]))  # peut importe la piece qu'on donne

        else:  # si on ne gagne pas
            return (action_contrainte(config, etatmax))


def empirique(config):
    return (ia(config, True))


def antijoueur(config):
    return (ia(config, False))


p = 2
staderech = 9


def minmax(config):
    """
    ---Entrée---
    config : configuration
    ---Sortie--- O(p^(6 + 2*profondeur))
    IA qui maximise ou minimise la valeur (valeur_minmax) du coup 
    méthode de Minmax de paramètres
        -profondeur (profondeur maximale de recherche)
        -staderech (stade à partir duquel on recherche)
        et d'heuristique d'évaluation heuristique
    Renvoie une configuration après un coup
    """
    stade = 15 - len(config[1])

    if stade < staderech or stade >= 15:  # tant qu'on choisit de ne pas rechercher
        return empirique(config)
    else:
        case, piece = decision_minmax(config, p)
        return jouer_coup(config, case, piece)


stadeprev = 7
longprev = 5


def montecarlo(config):
    """
    ---Entrée---
    config : configuration (plateau,stock,donnee)
    ---Sortie--- O(p^6 * longprev)
    IA qui maximise ou minimise la note (calcule_note) du coup 
    méthode de Monte Carlo de paramètres
        -longprev (nombre de parties continuées après chaque coup)
        -stadeprev (stade à partir duquel on prévoit)
        et d'heuristique de parcours Empirique
    Renvoie une configuration après un coup
    """
    stade = 15 - len(config[1])

    if stade < stadeprev or stade >= 15:  # tant qu'on choisit de ne pas prévoir
        return (empirique(config))
    else:
        return (prevoyance(config, True))


prevoyante = montecarlo


###########
##  Jeu  ##
###########

def jeu(config, joueur1, joueur2):
    """
    ---Entrée---
    config : configuration (plateau,stock,donnee)
    joueur1 (ou 2) : IA qui à une configuration revoie une configuration, et qui doit
        -renvoyer Vide s'il n'y a plus de pièces à donner
        -ne pas écraser une piece déjà placée
        -mettre en jeu une pièce du stock
        -supprimer la pièce mise en jeu du stock
    ---Sortie--- O(p^3 + p*C(joueur1) + p*C(joueur2))
    gagnant : numéro du joueur gagnant
    config : configuration terminale
    """
    stade = 15 - len(config[1])  # stade actuel

    while True:
        if est_fini(config[0]) != -1 or support(config[0]) == [] or config[2] == vide:
            break

        # qui joue ?
        if stade % 2 == 0:
            config = joueur1(config)
        else:
            config = joueur2(config)

        stade += 1

    gagnant = 0  # joueur gagnant, ex aequo si non gagné
    if est_gagne(config[0]):  # il existe un gagnant
        gagnant = (stade - 1) % 2 + 1

    return (gagnant, config)


####################
##  Statistiques  ##
####################

def temps_moyen(joueur, nomjoueur, N, parametres):
    """
    ---Entrée---
    joueur : IA qui à une config renvoie une config après un coup
    nomjoueur: string, nom de joueur
    N : nombre non nul de parties où joueur joue contre aléatoire
    parametres : liste de veleurs que peut prendre le paramètre global relatif à joueur
    ---Sortie--- O(N * (p^3 + (p - debut)*C(joueur))) si debut != p
    Renvoie (temps_exe,temps_exe_spec,temps_part,vict,nul) les temps moyens
        d'exécution de joueur pour les stades supérieurs (ou égaux) à debut
        et le pourcentage de réussite et de matchs nuls
    Affiche les valeurs calculées toutes les dix parties
    """
    global longprev, stadeprev

    temps_exe = np.zeros((len(parametres), 16, 2))  # [temps d'execution, nombre d'exécution]
    vict = np.zeros(3)  # pourcentage de réussite
    nul = np.zeros(3)  # pourcentage de matchs nuls

    for indparam in range(len(parametres)):
        print(1, end=' ')
        stadeprev = 5 + (parametres[indparam] / 5) // 2
        longprev = parametres[indparam]
        for indpart in range(0, N):
            if indpart % (N / 10) == 0:
                print(0, end=' ')
                # affiche le chargement

            # gestion du premier joueur
            prio = random.randint(0, 1)  # prio = 0 <=> c'est la référence qui commence

            config = config_initiale()
            stade = 0  # stade initial

            while True:
                if est_fini(config[0]) != -1 or support(config[0]) == [] or config[2] == vide:
                    break

                if stade % 2 == prio:  # tour de Empirique
                    config = empirique(config)
                else:
                    temps = time()  # début du coup
                    config = joueur(config)
                    temps_exe[indparam, stade, 0] += time() - temps
                    temps_exe[indparam, stade, 1] += 1

                stade += 1

            fin = est_fini(config[0])
            if fin == 1:  # s'il existe un gagnant
                gagnant = (stade - 1) % 2 + 1
                if (gagnant == 1 and prio == 1) or (gagnant == 2 and prio == 0):  # si le joueur gagne
                    vict[indparam] += 1
            elif fin == 0:
                nul[indparam] += 1

    vict, nul = vict * 100 / N, nul * 100 / N
    temps_exe[:, :, 0] = temps_exe[:, :, 0] / temps_exe[:, :, 1]

    plt.figure()
    for indparam in range(len(parametres)):
        chaine = "LongPrev = " + str(parametres[indparam]) + " : "
        chaine += str(vict[indparam]) + "% de victoires et " + str(nul[indparam]) + "% de matchs nuls"
        plt.plot(temps_exe[indparam, :, 0], label=chaine, linewidth=3)
    plt.plot([10 for i in range(16)], color='red', linewidth=3)
    plt.legend(fontsize=20, loc=1)

    chaine = "Resultat de " + str(N) + " parties\n de " + nomjoueur + " contre Empirique"
    plt.title(chaine, fontsize=25, fontweight='bold')  # titre

    plt.ylabel("Temps d'une execution (s)", fontsize=20, fontweight='bold', labelpad=20)  # axe
    plt.xlabel("Stade", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)  # graduations
    plt.grid(True)
    plt.show()


def statistiques(joueur, ref, N):
    """
    ---Entrée---
    joueur : IA qui à une config renvoie une config après un coup
    N : nombre non nul de parties où joueur joue contre aléatoire
    ---Sortie--- O(N * (p^3 + p*C(joueur) + p*C(ref)))
    Renvoie stats : liste de N valeurs du couple (vainc, stade)
        le vainqueur : 1 joueur, -1 ref, 0 ex aequo
        stade où la partie s'est arrêtée
    Affiche une barre de progression
    """
    stats = np.zeros((N, 2))

    for indpart in range(N):
        # if indpart % (N / 10) == 0:
        print(0, end=' ')

        # gestion du premier joueur
        prio = random.randint(0, 1)  # prio = 0 <=> c'est la référence qui commence

        vainc = 0

        config = config_initiale()
        stade = 0  # stade initial

        while True:
            if est_fini(config[0]) != -1 or support(config[0]) == [] or config[2] == vide:
                break

            if stade % 2 == prio:  # qui joue ?
                config = ref(config)
            else:
                config = joueur(config)

            stade += 1

        fin = est_fini(config[0])

        if fin == 1:  # s'il existe un gagnant
            gagnant = (stade - 1) % 2 + 1
            if (gagnant == 1 and prio == 1) or (gagnant == 2 and prio == 0):  # si le joueur étudié gagne
                vainc = 1
            else:
                vainc = -1
        else:
            vainc = 0

        stats[indpart] = np.array((vainc, stade))

    return (stats)


def comparaison_par_stade(joueur, nomjoueur, ref, nomref, N):
    """
    ---Entrée---
    joueur : joueur (IA) étudiée
    nomjoueur: string, nom de joueur
    ref : joueur référence
    nomref : string, nom de ref
    N : nombre de jeux réalisé
    ---Sortie--- O(N * (p^3 + p*C(joueur) + p*C(ref)))
    Affiche le graphique de l'évolution du pourcentage de réussite de joueur
        lors de parties contre ref
    Renvoie le stade moyen où joueur pose la première contrainte
    """
    stats = statistiques(joueur, ref, N)

    ###Nombre de victoire par stade
    compte = np.zeros((17, 3))  # pour un stade : [nbvict de ref, nbnul, nbdef de ref]
    ind = np.arange(17)  # indices
    valeurs, nombre = np.unique([17 * c[0] + c[1] for c in stats[:, 0:2]], return_counts=True)
    # on encode le couple (vainc,stade) par un entier à cause du fonctionnement de la fonction unique
    for i, v in enumerate(valeurs):
        v = int(v)
        compte[v % 17, (v // 17) + 1] = nombre[i]

    plt.figure()
    victoire = plt.bar(ind, compte[:, 2], 0.8, color='r')  # histogramme des victoires
    nul = plt.bar(ind, compte[:, 1], 0.8, color='k', bottom=compte[:, 2])
    defaite = plt.bar(ind, compte[:, 0], 0.8, color='b', bottom=compte[:, 2] + compte[:, 1])
    plt.axis([3, 17, 0, 0.7 * N])
    plt.tight_layout(pad=5)  # marges

    chaine = "Resultat de " + str(N) + " parties\n de " + nomjoueur + " contre " + nomref
    plt.title(chaine, fontsize=23, fontweight='bold')  # titre

    chainevict = "Stade moyen de victoire de " + nomjoueur + " : "
    chainevict += str(round(np.average(np.arange(17), weights=compte[:, 2]), 1))
    chainenul = "Pourcentage de matchs nuls  : " + str(compte[16, 1] * 100 / N) + "%"
    chainedef = "Stade moyen de victoire de " + nomref + " : "
    chainedef += str(round(np.average(np.arange(17), weights=compte[:, 0]), 1))
    plt.legend((victoire, nul, defaite), (chainevict, chainenul, chainedef), framealpha=0.75,
               loc='upper left')  # légende

    plt.ylabel("Nombre de victoire", fontsize=20, fontweight='bold')  # axes
    plt.xlabel("Stade du jeu", fontsize=20, fontweight='bold')
    plt.xticks(np.linspace(3.5, 16.5, 14), np.arange(3, 17), fontsize=15)  # graduation
    plt.yticks(fontsize=15)
    plt.grid(True, axis='y')
    plt.show()


def comparaison(joueur, nomjoueur, ref, nomref, N):
    """
    ---Entrée---
    joueur : joueur (IA) étudiée
    nomjoueur: string, nom de joueur
    ref : joueur référence
    nomref : string, nom de ref
    N : nombre de jeux réalisé
    ---Sortie--- O(N * (p^3 + p*C(joueur) + p*C(ref))) Empirique-Minmax(Sr=11,P=4) en 10min
    Affiche le graphique du nombre de victoire de joueur, ref ou matchs nuls
    """
    stats = statistiques(joueur, ref, N)

    ###Pourcentage de réussite
    valeurs, nombre = np.unique(stats[:, 0], return_counts=True)
    compte = np.array([0, 0, 0])  # [nombre de victoires de ref, nombre de matchs nuls, nombre de défaites de ref]
    for i, v in enumerate(valeurs):
        compte[int(v) + 1] = nombre[i]

    plt.figure()
    n0, bins0, patches0 = plt.hist(stats[:, 0], facecolor='black')
    plt.axis([-1.5, 1.5, 0, N])
    plt.tight_layout(pad=5)  # marges

    plt.text(0.75, compte[2] + N / 50, str(compte[2] * 100 / N) + '%', fontsize=23, fontweight='bold')
    plt.text(-0.05, compte[1] + N / 50, str(compte[1] * 100 / N) + '%', fontsize=23, fontweight='bold')
    plt.text(-1.05, compte[0] + N / 50, str(compte[0] * 100 / N) + '%', fontsize=23, fontweight='bold')

    chaine = 'Resultat de ' + str(N) + ' parties\n de ' + nomjoueur + ' contre ' + nomref
    plt.title(chaine, fontsize=25, fontweight='bold')  # titre

    plt.ylabel('Nombre de parties gagnees', fontsize=20)  # axe
    plt.xticks([-1, 0, 1], (nomref, 'Match nul', nomjoueur), fontsize=20, fontweight='bold')
    plt.yticks(fontsize=15)  # graduations
    plt.grid(True, axis='y')
    plt.show()


def nuls(N):
    """
    ---Entrée---
    ---Sortie--- O(p!*p)
    Renvoie une approximation du pourcentage de matchs nuls du Quarto
    """
    plateau, stock, donnee = config_initiale()
    stock.append(donnee)
    nul = 0

    for i in range(N):
        if i % (N / 10) == 0:
            print(nul / (i + 1) * 100)

        sigma = random.shuffle(stock)
        for j, piece in enumerate(sigma):
            plateau[j // 4][j % 4] = piece
            if not est_gagne(plateau):
                nul += 1

    return (nul / N * 100)


# for _ in range(10):
#     config = config_initiale()
#     for _ in range(8):
#         config = aleatoire(config)
#
#     p = 2
#     print(minimax_s(config, p))
#     print(valeur_minmax(config, False, (-1, 0, 0), (1, 16, 480), p))
#     print(valeur_m(config, p))
#     print("########################")
#     # heuristique(config, False)

# l = statistiques(minmax, empirique, 10)
# np.save("fichier_marrant_1.npy", l)

comparaison(minmax, "minmax", empirique, "empirique", 10)
