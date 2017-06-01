# -*- coding: utf-8 -*-
from random import randint
import pygame as pg
from pygame.locals import *
import numpy as np

from main import config_initiale, aleatoire, empirique, minmax, est_fini, support, copie_config, removearray, \
    heuristique, alignements

vide = np.zeros(4)

def est_vide(l):
    return np.array_equal(np.zeros(4), l)

# On note p = 16 le nombre de cases, pièces, et stade maximaux

#############################
##  Initialisation pygame  ##
#############################

pg.init()  # initialise
fenetre = pg.display.set_mode((860, 510))

P0000 = pg.image.load("/home/victor/Quarto/images/pieces/P0000.png").convert()
P0001 = pg.image.load("/home/victor/Quarto/images/pieces/P0001.png").convert()
P0010 = pg.image.load("/home/victor/Quarto/images/pieces/P0010.png").convert()
P0011 = pg.image.load("/home/victor/Quarto/images/pieces/P0011.png").convert()
P0100 = pg.image.load("/home/victor/Quarto/images/pieces/P0100.png").convert()
P0101 = pg.image.load("/home/victor/Quarto/images/pieces/P0101.png").convert()
P0110 = pg.image.load("/home/victor/Quarto/images/pieces/P0110.png").convert()
P0111 = pg.image.load("/home/victor/Quarto/images/pieces/P0111.png").convert()
P1000 = pg.image.load("/home/victor/Quarto/images/pieces/P1000.png").convert()
P1001 = pg.image.load("/home/victor/Quarto/images/pieces/P1001.png").convert()
P1010 = pg.image.load("/home/victor/Quarto/images/pieces/P1010.png").convert()
P1011 = pg.image.load("/home/victor/Quarto/images/pieces/P1011.png").convert()
P1100 = pg.image.load("/home/victor/Quarto/images/pieces/P1100.png").convert()
P1101 = pg.image.load("/home/victor/Quarto/images/pieces/P1101.png").convert()
P1110 = pg.image.load("/home/victor/Quarto/images/pieces/P1110.png").convert()
P1111 = pg.image.load("/home/victor/Quarto/images/pieces/P1111.png").convert()

joueur1 = pg.image.load("/home/victor/Quarto/images/Joueur 1.png").convert()
joueur2 = pg.image.load("/home/victor/Quarto/images/Joueur 2.png").convert()
placez = pg.image.load("/home/victor/Quarto/images/Placez.png").convert()
donnez = pg.image.load("/home/victor/Quarto/images/Donnez.png").convert()
patientez = pg.image.load("/home/victor/Quarto/images/Patientez.png").convert()
gagnant1 = pg.image.load("/home/victor/Quarto/images/Gagnant1.png").convert()
gagnant2 = pg.image.load("/home/victor/Quarto/images/Gagnant2.png").convert()
matchnul = pg.image.load("/home/victor/Quarto/images/Matchnul.png").convert()
toaccueil = pg.image.load("/home/victor/Quarto/images/Toaccueil.png").convert()
fond = pg.image.load("/home/victor/Quarto/images/Fond.png").convert()
accueil = pg.image.load("/home/victor/Quarto/images/Accueil.png").convert()

fenetre.blit(fond, (0, 0))  # empile
pg.display.flip()  # actualise

############################
##   Gestion de l'Ecran   ##
############################

images = [fond, accueil,
          P0000, P0001, P0010, P0011,
          P0100, P0101, P0110, P0111,
          P1000, P1001, P1010, P1011,
          P1100, P1101, P1110, P1111,
          joueur1, joueur2,
          placez, donnez, patientez,
          matchnul, gagnant1, gagnant2,
          toaccueil]

position_image = [(0, 0), (0, 0),
                  (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1), (-1, -1), (-1, -1),
                  (-1, -1)]
# position en cours des pièces

dico_plateau = [
    [
        [[(24 * k + 5) * 5, (24 * (k + 1) + 1) * 5],
         [(24 * l + 5) * 5, (24 * (l + 1) + 1) * 5]] for k in range(4)]
    for l in range(4)]


# coordonnées des cases sur le plateau sous la forme [ [x_min, x_max],
#                                                   [y_min, y_max] ]

def up(n):  # permet d'espacer les images
    return int(n >= 2)


dico_stock = [
    [
        [[(17 * k + 102 + up(k)) * 5, (17 * (k + 1) + 100 + up(k)) * 5],
         [(17 * l + 32 + up(l)) * 5, (17 * (l + 1) + 30 + up(l)) * 5]] for k in range(4)]
    for l in range(4)]
# coordonnées des pieces du stock

dico_accueil = [
    [
        [[(56 * k + 6) * 5, (56 * (k + 1) - 2) * 5],
         [(18 * l + 38) * 5, (18 * (l + 1) + 30) * 5]] for k in range(3)]
    for l in range(2)]

dico_donnee = [[127 * 5, 147 * 5],
               [4 * 5, 24 * 5]]


# coordonees de la donnee

def affichage():
    """
    ---Entrée---
    
    ---Sortie--- O(p)
    Empile les images sur la fenêtre (la rafraichit)
    les images de coordonnées (-1,-1) ne sont pas affichées
    L'ordre d'affichage est celui de la liste images
    """
    global images, position_image

    for indim in range(len(images)):
        coord = position_image[indim]
        if coord != (-1, -1):
            fenetre.blit(images[indim], coord)


##############################
##  Fonctions de conversion ##
##############################

def coord_to_inddico(x, y, dico):
    """
    ---Entrée---
    x, y : les coordonnees du clic
    dico : matrice de coordonnees sous la forme [ [x_min, x_max],
                                                  [y_min, y_max] ]
        (dico_donnee ne convient pas)
    ---Sortie--- O(dim(dico))
    Renvoie le couple d'indice de l'objet de dico sélectionné
        Si ne corespond pas à des indices valables, renvoie (-1, -1)
    """
    for k in range(len(dico)):
        for l in range(len(dico[0])):

            X, Y = dico[k][l][0], dico[k][l][1]
            if X[0] <= x <= X[1] and Y[0] <= y <= Y[1]:
                # a-t-on cliqué dans la zône délimitée par dico[k][l] ?
                return (k, l)

    return (-1, -1)  # on a cliqué autre part


def piece_to_indim(piece):
    """
    ---Entrée---
    piece : pièce
    ---Sortie--- O(1)
    Renvoie l'indice de la piece dans la liste des images
    """
    piece = (piece + 1) // 2
    return piece[0] * 8 + piece[1] * 4 + piece[2] * 2 + piece[3] + 2  # +2 car apres le fond et accueil


def inddico_to_joueur(k, l):
    """
    ---Entrée---
    k,l : indices du joueur selectionné dans le menu de l'accueil
    ---Sortie--- O(1)
    Renvoie le joueur sélectionné, ou 0 si "2 joueurs"
    """
    if k == 0:  # ligne
        if l == 0:  # colonne
            return aleatoire
        elif l == 2:
            return empirique
    else:
        if l == 0:
            return minmax
        if l == 1:
            return (prevoyante)
        else:
            return (antijoueur)

    return (0)


def matrice_stock(stock):
    """
    ---Entrée---
    
    ---Sortie--- O(p)
    Renvoie le stock ordonné en matrice selon dico_stock
        Contient vide si la pièce n'est pas dans le stock
    """
    mat = []
    # initialisation de la matrice
    for i in range(4):
        ligne = []
        for j in range(4):
            ligne.append(vide)
        mat.append(ligne)

    for piece in stock:  # on place dans la matrice les pieces du stock
        piece = (piece + 1) // 2
        k = piece[0] * 2 + piece[1]
        l = piece[2] * 2 + piece[3]
        mat[k][l] = piece

    return (mat)


def modifications(config1, config2):
    """
    ---Entrée---
    config1,config2 : configuration où un coup a été joué de 1 à 2
    ---Sortie--- O(p)
    Renvoie la case sur laquelle a été placée la donnee1, et la pièce donnee2
    """
    plateau1, plateau2 = config1[0], config2[0]

    for i in range(4):
        for j in range(4):  # indices de la case
            if not np.array_equal(plateau1[i, j], plateau2[i, j]):
                return (i, j), config2[2]


#############################
##  Changements de phases  ##
#############################

def initialise_plateau():
    """
    ---Entrée---
    
    ---Sortie--- O(p)
    Modifie les images à afficher pour débuter une partie
    """
    global position_image, dico_stock

    for k in range(4):  # initialisation des pièces
        for l in range(4):
            x = dico_stock[k][l][0][0]  # coordonnées des pièces du stock sur la fenêtre
            y = dico_stock[k][l][1][0]
            position_image[4 * k + l + 2] = (x, y)

    position_image[2] = (129 * 5, 30)  # donnee initiale
    position_image[18] = (510, 135)  # joueur1
    position_image[20] = (510, 10)  # placez


def changement_joueur(stade):
    """
    ---Entrée---
    stade : numero du stade actuel
    ---Sortie--- O(1)
    Modifie les images à afficher suite à un changement de joueur
    """
    if stade % 2 == 1:  # si au tour du joueur 2
        position_image[18] = (-1, -1)
        position_image[19] = (510, 135)
    else:
        position_image[19] = (-1, -1)
        position_image[18] = (510, 135)


def changement_action(action):
    """
    ---Entrée---
    action : type de coup actuel
        *0 : Placement (on doit placer la donnee)
        *1 : Don (on doit choisir la donnee)
    ---Sortie--- O(1)
    Modifie les images à afficher suite à un changement d'action
    """
    if action == 0:  # si on doit placer
        position_image[20] = (510, 10)
        position_image[21] = (-1, -1)
    else:
        position_image[20] = (-1, -1)
        position_image[21] = (745, 10)


def placement(k, l, plateau, donnee):
    """
    ---Entrée---
    k,l : coordonnées de la case du placement
    plateau : plateau après placement
    donnee : donnée  en cours
    ---Sortie--- O(p^2)
    Modifie les images à afficher et les paramètres suite au placement
    """
    global position_image, action, fin

    x = dico_plateau[k][l][0][0] + 12  # +12 pour centrer
    y = dico_plateau[k][l][1][0] + 12
    position_image[piece_to_indim(donnee)] = (x, y)  # on colle la donnee à l'emplacement choisi

    action = 1
    changement_action(action)
    fin = est_fini(plateau)  # le jeu est-il fini ?


def fini(fin, stade):
    """
    ---Entrée---
    fin : * 1 s'il y a un gagnant
          * 0 si match nul
    stade : stade actuel
    ---Sortie--- O(1)
    Modifie les images à afficher et les paramètres en fin de partie
    """
    global position_image

    if fin == 0:
        gagnant = 0  # match nul
    elif fin == 1:
        gagnant = stade % 2 + 1
    position_image[21] = (-1, -1)  # on n'affiche plus les actions à faire
    position_image[20] = (-1, -1)
    position_image[22] = (-1, -1)
    position_image[23 + gagnant] = (510, 135)  # on affiche le gagnant
    position_image[26] = (635, 20)  # bouton retour a l'accueil


def don(piece):
    """
    ---Entrée---
    piece : nouvelle donnée
    ---Sortie--- O(1)
    Modifie les images à afficher et les paramètres suite au don
    """
    global position_image, action, stade

    x = dico_donnee[0][0] + 12  # +12 pour centrer
    y = dico_donnee[1][0] + 12
    position_image[piece_to_indim(piece)] = (x, y)  # on colle la pièce à l'emplacement de la donnee

    action = 0
    stade += 1  # on passe a l'autre joueur
    changement_action(action)
    changement_joueur(stade)


def to_accueil():
    """
    ---Entrée---
    ---Sortie--- O(p)
    Modifie les images à afficher pour retourner à l'accueil
    """
    global position_image

    for i in range(2, len(images)):
        position_image[i] = (-1, -1)

    position_image[1] = (0, 0)  # accueil


#################
##  Affichage  ##
#################

### Initialisation
config = config_initiale()
continuer = True  # bolléen principal

### Paramètres d'état
home = True  # à l'accueil
x, y = 0, 0  # coordonnées du dernier clic

prio = 0  # prio = 0 <=> c'est l'humain qui commence
fin = -1  # fin = * 1 s'il y a un gagnant
#               * 0 si match nul
#               * -1 si indéterminé
stade = 0  # stade actuel

action = 0  # action = * 0 : Placement (on doit placer la donnee)
#                    * 1 : Don (on doit choisir la donnee)

while continuer:
    """O(p^3 + p*C(joueur) + p*C(humain))"""
    for event in pg.event.get():

        ### Arrêt
        if event.type == QUIT:
            continuer = False

        ### Clic
        elif event.type == MOUSEBUTTONDOWN:
            x = event.pos[0]
            y = event.pos[1]

        ### Accueil
        if home:
            """O(p)"""
            k, l = coord_to_inddico(x, y, dico_accueil)
            if k != -1 and l != -1:  # si clic correct
                x, y = -1, -1  # réinitialisation du clic
                position_image[1] = (-1, -1)  # retrait de l'image accueil
                home = False

                initialise_plateau()
                joueur = inddico_to_joueur(k, l)
                prio = randint(0, 1)  # détermination au hasard de celui qui commence

        ### Jeu fini
        elif fin != -1:
            """O(p)"""
            X = dico_donnee[0]
            Y = dico_donnee[1]

            if X[0] <= x <= X[1] and Y[0] <= y <= Y[1]:  # si on souhaîte retourner à l'accueil
                to_accueil()  # réinitialisation de la fenetre
                config = config_initiale()  # du quarto
                # et des parametres
                home = True
                joueur = 0
                prio = 0
                fin = -1
                action = 0
                stade = 0

        ### Humain
        elif joueur == 0 or (joueur != 0 and prio == stade % 2):
            """O(p^2 + C(humain))"""
            plateau, stock, donnee = config

            ### Placement
            if action == 0:
                k, l = coord_to_inddico(x, y, dico_plateau)
                if k != -1 and l != -1 and (k, l) in support(plateau):
                    # si l'humain a cliqué sur une case correcte non prise
                    plateau[k][l] = donnee  # on pose la piece
                    placement(k, l, plateau, donnee)

                    if fin != -1:  # si le jeu est fini
                        donnee = vide
                        fini(fin, stade)

            ### Don
            else:
                k, l = coord_to_inddico(x, y, dico_stock)
                if k != -1 and l != -1:
                    piece = matrice_stock(stock)[k][l]
                    if not est_vide(piece):
                        # si l'humain a cliqué sur une pièce encore dans le stock
                        donnee = piece * 2 - 1
                        removearray(stock, donnee)
                        don(donnee)

            config = (plateau, stock, donnee)

        ### Intelligence artificielle
        elif joueur != 0 and prio != stade % 2:
            """O(p^2 + C(joueur))"""
            position_image[22] = (510, 10)  # patientez
            affichage()
            pg.display.flip()

            copie = copie_config(config)
            config = joueur(config)
            case, piece = modifications(copie, config)

            ### Placement
            k, l = case
            placement(k, l, config[0], copie[2])

            ### Don
            if not est_vide(piece):
                don(piece)

            if fin != -1:
                fini(fin, stade)
            position_image[22] = (-1, -1)

    # print(heuristique(config))
    # print(config[0])
    # print(np.sum(alignements(config[0]), axis=1))
    affichage()
    pg.display.flip()

print("Fin")
