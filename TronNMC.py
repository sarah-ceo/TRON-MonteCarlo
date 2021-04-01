import random
import time
import copy
import numpy
import numba
from numba import jit

## fenetre d'affichage

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
plt.ion()
plt.show()
fig,axes = plt.subplots(1,1)
fig.canvas.set_window_title('TRON')


#################################################################################
#
#  Parametres du jeu

LARGEUR = 13
HAUTEUR = 17
L = 20  # largeur d'une case du jeu en pixel

canvas = None   # zone de dessin
Grille = None   # grille du jeu
posJ1  = None   # position du joueur 1 (x,y)
NbPartie = 0   # Nombre de parties effectuées
Scores = [0 , 0]  # score de la partie / total des scores des differentes parties

def InitPartie():  
    global Grille, PosJ1, NbPartie, Scores
    
    NbPartie += 1
    Scores[0] = 0
    
    Grille = numpy.zeros((LARGEUR, HAUTEUR))
    
    # #positionne les murs de l'arene
    for x in range(LARGEUR):
       Grille[x][0] = 10
       Grille[x][HAUTEUR-1] = 10
       for y in range(HAUTEUR):
           if x>= LARGEUR/3-1 and x<= 2*LARGEUR/3:
               if y>= HAUTEUR/3 and y<= 2*HAUTEUR/3:
                   if y%2==0:
                       Grille[x][y] = 10
       
       
    for y in range(HAUTEUR):
       Grille[LARGEUR-1][y] = 10
       Grille[0][y] = 10
    
    # position du joueur 1
    PosJ1 = (LARGEUR//2,1)


#################################################################################
#
# gestion du joueur humain et de l'IA
    
@jit(nopython=True)
def availablePositions(grille, x, y):
    positions = numpy.zeros(0).reshape(0, 2).astype(numpy.int32)
    for i in range(-1,2,2):
        if grille[x+i][y]==0:
            positions = numpy.vstack((positions, numpy.array((i, 0)).reshape(1,2).astype(numpy.int32)))
    for j in range(-1,2,2):
        if grille[x][y+j]==0:
            positions = numpy.vstack((positions, numpy.array((0, j)).reshape(1,2).astype(numpy.int32)))
    return positions
    
@jit
def SimulationPartie (GrilleTemp,x,y):
    nb_cases = 0
    while True:
        L = availablePositions(GrilleTemp,x,y)
        if len(L) == 0:
            return nb_cases
        direction = L[random.randrange(len(L))]
        GrilleTemp[x][y] = 1
        x+=direction[0]
        y+=direction[1]
        nb_cases +=1

@jit        
def MonteCarlo(Grille,x,y,nombreParties):
    Total = 0
    for i in range(nombreParties):
        Grille2 = numpy.copy(Grille)
        Total += SimulationPartie(Grille2,x,y)
    return Total

@jit(nopython=True)
def NMC(grille, level, posJ):
    GrilleTemp = numpy.copy(grille)
    L = availablePositions(GrilleTemp,posJ[0],posJ[1])
    sequence = numpy.zeros(0).reshape(0, 2).astype(numpy.int32)
    if level == 0:
        nb_cases = 0
        while L.size != 0:
            direction = L[random.randrange(len(L))]
            posJ = (posJ[0]+direction[0], posJ[1]+direction[1])
            sequence = numpy.vstack((sequence, direction.reshape(1, 2)))
            GrilleTemp[posJ[0]][posJ[1]] = 1
            L = availablePositions(GrilleTemp,posJ[0],posJ[1])
            nb_cases +=1
        return nb_cases, sequence
    else:
        temp_best_score = -1
        best_score = 0
        numcoup = 0
        while L.size != 0:
            for i in range(L.shape[0]):
                coup_possible = L[i]
                position = (posJ[0]+coup_possible[0], posJ[1]+coup_possible[1])
                GrilleTemp[position[0]][position[1]] = 1
                score, sequence_associee = NMC(GrilleTemp, level-1, position)
                GrilleTemp[position[0]][position[1]] = 0
                if score > temp_best_score:
                    best_score += score - temp_best_score
                    temp_best_score=score
                    sequence = numpy.vstack((sequence[0:numcoup], numpy.vstack((coup_possible.reshape(1,2), sequence_associee))))
            node = sequence[numcoup]
            posJ = (posJ[0]+node[0], posJ[1]+node[1])
            GrilleTemp[posJ[0]][posJ[1]] = 1
            L = availablePositions(GrilleTemp,posJ[0],posJ[1])
            temp_best_score-=1
            numcoup +=1
        return best_score, sequence
            
def Play():   
    global Scores
    Tstart = time.time()
    global  PosJ1        
    Grille[PosJ1[0]][PosJ1[1]] = 1 # laisse la trace de la moto
    best_score, sequence = NMC(Grille, 2, PosJ1)
    #print(time.time()-Tstart)
    for coup in sequence:
        Grille[PosJ1[0]][PosJ1[1]] = 1
        PosJ1 = (PosJ1[0]+coup[0], PosJ1[1]+coup[1])
    # fin de traitement
    
        Scores[0] += 1
        Affiche()
    
    # detection de la collision
       
   
    
    
################################################################################
#    
# Dessine la grille de jeu


def Affiche():
    axes.clear()
    
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.axis('off')
    fig.patch.set_facecolor((0,0,0))
    
    axes.set_aspect(1)
    
    # dessin des murs

    Murs  = []
    Bords = []
    for x in range (LARGEUR):
       for y in range (HAUTEUR):
           if ( Grille[x][y] == 10 ) : Bords.append(  plt.Rectangle((x,y), width = 1, height = 1 ) )
           if ( Grille[x][y] == 1  ) : Murs.append(  plt.Rectangle((x,y), width = 1, height = 1 ) )
        
    axes.add_collection (  matplotlib.collections.PatchCollection(Murs,   facecolors = (1.0, 0.47, 0.42)) )
    axes.add_collection (  matplotlib.collections.PatchCollection(Bords,  facecolors = (0.6, 0.6, 0.6)) )
    
    # dessin de la moto
  
    axes.add_patch(plt.Circle((PosJ1[0]+0.5,PosJ1[1]+0.5), radius= 0.5, facecolor = (1.0, 0, 0) ))
    
    # demande reaffichage
    fig.canvas.draw()
    fig.canvas.flush_events()  
 

################################################################################
#    
# Lancement des parties      
          
def GestionnaireDeParties():
    global Scores
   
    for i in range(3):
        time.sleep(1) # pause dune seconde entre chaque partie
        InitPartie()
        Play()
        Scores[1] += Scores[0]   # total des scores des parties
        Affiche()
        ScoMoyen = Scores[1]//(i+1)
        print("Partie " + str(i+1) + " === Score : " + str(Scores[0]) + " === Moy " + str(ScoMoyen) )
        
     
GestionnaireDeParties()

  


    
        

      
 

