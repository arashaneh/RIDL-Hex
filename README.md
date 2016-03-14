# RIDL-HexD
======
X is an efficient sparse dictionary learning algorithm for extracting rotation-aware or rotation-invariant features from images.
X benefits from a novel hexagonal sampling grid, a modified rotation-invariant sparse coding (based on  Orthogonal Matching Pursuit) and a rotation learning algorithm, based on an exponentiated-gradient algorithm, for updating the dictionary matrix. 
The dictionary elements in X are parametrized by their orientation, and a wide range of patterns can be well represented with a small dictionary size. X achieves faster convergence than other dictionary learning algorithms not using this type of parameterization. 


Requirements
======
This implementation of X is based on python 2.7 and requires following python packages:

Main dependencies:
- numpy 
- scipy 
- matplotlib

and limited usage of 
- opencv (in order to load image files)
- pillow (used in creating a circular mask for square grid)
- scikit-learn (used in efficient shuffling of the data matrix)


important note
-------
Since calculating norms of columns of a matrix is a feature added in numpy 1.8, you need to have numpy 1.8 or newer. 
This code is tested on:
 
- scipy (0.12.1)
- numpy (1.10.4)
- opencv (2.4.5)
- pillow (2.0.)
- matplotlib (1.2.0)
- scikit-learan (0.17)


Examples
======


*  *foveated_sampling.py* creates a foveated samples from an image and displays the hexagonal grid structure used in foveated sampling

*  *dict_learning_comp_sgd.py* compares the convergence of reconstruction error for dictionary learning using *stochastic gradient descent* for rotation invariant and dictionary matrix with independent elements for different grid structures.

* *dict_learning_comp_rotation_update.py* compares the convergence of reconstruction error for dictionary learning using *rotation learning* for rotation invariant and dictionary matrix with independent elements for different grid strctures.

* *activation_patterns_dict_elements.py* learns a rotation invariant dictionary for three images (Lena, Barbara and a Van Gogh painting), and draws the pattern of usage of different orientations of each dictionary element.


