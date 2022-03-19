#Running our code
Our driver code can be found in the omr.py file, and can be run by calling ./omr.py image.png



# OCR.py

## Part 3
### Helper functions: 
get_value_matrix(Image *pImage*, tuple *pPosition*, array *pKernel*): 
>describe function

compute_conv_val(array *pValues*, array *pKernel*):
>describe function

convolve_brute_force(Image *pI*, array *pH*): 
>describe function

get_value_matrix_RGB(Image *pImage*, tuple *pPosition*, array *pKernel*): 
>describe function


### Main Function: 
convolve_brute_force_RGB
> describe function here




## Part 4

### Helper functions: 
get_column_vectors (2d-array *kernel*):
>breaks down a kernel into its Hx and Hy vectors so that we can convolve along the axis individually. 

get_grayscale_matrix(Image *image*):
>takes an image object as an input and returns a 2d array that is a matrix representation of the image.
>The values are the grayscale values for each individual pixel.  The indicies are the pixels (x,y) position. 

### Main Function: 
convolve_grayscale(Image *image*, Kernel/2d-array *H*): 
> Gets the value matrix for the image file so that we have the grayscale values. <br>
> Breaks the kernel down into its two column vectors. <br>
> We convolve horizontally across the value matrix with the Hx column vector <br>
> We then transpose the new matrix in order to convolve horizontally again with the Hy vector<br>
> Because we transposed this is the same as convolving vertically. <br>
> We then transpose again to get back to the initial orientation of the value matrix. <br>
> We then use this value matrix to write a new grayscale image. <br>

## Part 5

### Helper Functions: 
compute_hamming_distance(array *value_matrix*, Image *template*): 
> describe function

### Main Function: 
locate_symbol(Image *image*, Image *template*): 
>describe function 


## Part 6

### Helper Functions: 
sobel(Image *image*): 
> takes a grayscale input and edge detects using sobel operators.<br>
> first convolving with the x direction, then the Y direction. 

dist(tuple *pPos1*, tuple *pPos2*):
>finds the euclidean distance between two points.

dFuncPreCompute(Image *pImage*): 
>This function performs the pre-computing for the dFunc

dFunc(int *i*, int *j*): 
>For every pixel, i,j, returns the distance (in pixels) to the closes edge pixel in I. 

tFunc(int *i*, int *j*): 
>returns 1 if pixel (i,j) is an edge, and 0 otherwise. 

### Main Function: 


## Part 7

### Main Function: 
hough_transform (Image *image*): 
>gets us a hough space where the x (row value) is every possible first line position. <br>
> the y (column value) is a possible spacing (with an upper bound at 20 pixels). <br> 



# driver.py
The file that uses all previously implemented functions in order to perform the operation required for the assignment, as well as evaluate our results. 

### Helper Function:
load_image(String *path*): 
>returns an image object from the given path

pick_templates(int *headsize*, int *quarter*, int *eighth*): 
>when given the sizes of the headsize, quarterrest size, and eighth rest size, <br>
>will return a list of the respectable templates.





