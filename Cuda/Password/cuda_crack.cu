#include <stdio.h>
#include <cuda_runtime_api.h>

/****************************************************************************
  This program gives an example of a poor way to implement a password cracker
  in CUDA C. It is poor because it acheives this with just one thread, which
  is obviously not good given the scale of parallelism available to CUDA
  programs.
  
  The intentions of this program are:
    1) Demonstrate the use of __device__ and __global__ functions
    2) Enable a simulation of password cracking in the absence of library 
       with equivalent functionality to libcrypt. The password to be found
       is hardcoded into a function called is_a_match.   

  Compile and run with:
    nvcc -o cuda_crack.cu cuda_crack
    ./cuda_crack
   
  Dr Kevan Buckley, University of Wolverhampton, 2018
*****************************************************************************/

/****************************************************************************
  This function returns 1 if the attempt at cracking the password is 
  identical to the plain text password string stored in the program. 
  Otherwise,it returns 0.
*****************************************************************************/
__device__ int is_a_match(char *attempt)
{
	char plain_password[] = "RG";
  
  char *a = attempt;
  char *p = plain_password;
  
  while(*a == *p)
	{
    if(*a == '\0')
		{
      return 1;
    }
    a++;
    p++;
  }
  return 0;
}

/****************************************************************************
  The kernel function assume that there will be only one thread and uses 
  nested loops to generate all possible passwords and test whether they match
  the hidden password.
*****************************************************************************/ 
__global__ void kernel()
{
  char i, j;		

	int threadID = threadIdx.x;
	int blockID = blockIdx.x;

	int tID = blockID * blockDim.x + threadID;

	// prints a unique thread id.
	printf("Thread id is %d\n", tID);
  
  char password[3];
  password[2] = '\0';
  
  for(i='A'; i<='Z'; i++)
	{
    password[0] = i;

    for(j='A'; j<='Z'; j++)
		{
      password[1] = j;

      if(is_a_match(password))
			{
        printf("password found: %s\n, Thread Id is %d\n threadID" , password, tID);
      } 
			else
			{
	     //printf("tried: %s\n", password);		  
      }
    }
  }
}

int main()
{	
	
	char arrayLetters[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
	'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};

	char *gpuLetters;

	//cudamalloc 
	cudaMalloc((void**) &gpuLetters, 26*sizeof(char));

	cudaMemcpy(arrayLetters, gpuLetters, 26*sizeof(char), cudaMemcpyHostToDevice);

	// kernel to launch the program, 26, 26 for 26 blocks and 26 threads;
  kernel <<<26, 26>>>();
  cudaThreadSynchronize();	

	//cudamemcpy
	cudaMemcpy(gpuLetters, arrayLetters, 26*sizeof(char), cudaMemcpyDeviceToHost);

  return 0;
}
