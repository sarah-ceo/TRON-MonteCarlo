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

@jit
def availablePositions(grille, x, y):
    positions = []
    for i in range(-1,2,2):
        if grille[x+i][y]==0:
            positions.append((i,0))
    for j in range(-1,2,2):
        if grille[x][y+j]==0:
            positions.append((0,j))
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

def Play():   
    global Scores
    
    while (True):   
      Tstart = time.time()
      global  PosJ1        
      Grille[PosJ1[0]][PosJ1[1]] = 1 # laisse la trace de la moto
      positions = availablePositions(Grille, PosJ1[0], PosJ1[1])
      if len(positions)>0:
        #random_index = random.randrange(len(positions))
        MCT_scores = []
        for i in range(len(positions)):
            MCT_scores.append(MonteCarlo(Grille, PosJ1[0]+positions[i][0], PosJ1[1]+positions[i][1], 10000))
        best_index = MCT_scores.index(max(MCT_scores))
        PosJ1 = ( PosJ1[0]+positions[best_index][0] ,  PosJ1[1]+positions[best_index][1])  #deplacement
      
      # fin de traitement
      
      Scores[0] +=1 
      #print(time.time()-Tstart)
      Affiche()
      
      # detection de la collision  
      
      if ( Grille[PosJ1[0]][PosJ1[1]] != 0 ): return  
       
   
    
    
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

  


    
        

      
 

