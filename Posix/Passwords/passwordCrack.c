#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <crypt.h>

/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer. Your personalised data set is included in the
  code.

  Compile with:
    cc -o CrackAZ99-With-Data CrackAZ99-With-Data.c -lcrypt

  If you want to analyse the results then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:

    ./CrackAZ99-With-Data > results.txt

  Dr Kevan Buckley, University of Wolverhampton, 2018
******************************************************************************/
int n_passwords = 4;

char *encrypted_passwords[] = 
{
  "$6$KB$SjQpmnMtmvzsiL43FPUsaQPwtGmR52870.sgY0BQQ/SEWOhg57FMPAqJ.DUJKR9QvsQD4KEqgHlTAMYScXwkA.",
  "$6$KB$dWRMunG0MNBi43C6.XgLgxVY8DL31iybaffxwt4nS465Yl8j1WVCE3yrG0r6OowrSN5Z6svImqyGLnWXaFcgr.",
  "$6$KB$QA4KTFfCfcu3nG9ZdsYoviipAUs281dawKFcDGg0395mzu61nkt.W9a3FHj9696Q7dMQdoIRUudaKpr87EYf3/",
  "$6$KB$RdoFybakl8y8pqldFdW4.I30Wtt1gBjGqSqO1Uuqrs9IESm5VsjzXULsxm9GzeylmjdCYPupiFe5jH0FqcVgb0"
};


struct structVars
{
	long long start;
	long long finish;
	char *encrypted_passwords;
};

/**
 Required by lack of standard function in C.  
*/

void substr(char *dest, char *src, int start, int length)
{
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

int time_difference(struct timespec *start, struct timespec *finish,
                               long long int *difference) 
{
   long long int ds =  finish->tv_sec - start->tv_sec;
   long long int dn =  finish->tv_nsec - start->tv_nsec;
 
   if(dn < 0 )
	 {
     ds--;
     dn += 1000000000;
   }
   *difference = ds * 1000000000 + dn;
   return !(*difference > 0);
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the
 start of the line. Note that one of the most time consuming operations that
 it performs is the output of intermediate results, so performance experiments
 for this kind of program should not include this. i.e. comment out the printfs.
*/

void *crack(void *arg)
{

	struct structVars *vars = (struct structVars*) arg;
	
  int x, y, z;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space for \0
  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt,vars->encrypted_passwords, 0, 6);

	char startOf = (char)(vars->finish);
	char finishOf = (char)(vars->finish);

	// start from struct, to end of struct. vars->
  for(x = startOf; x <= finishOf; x++)
	{ 
    for(y = startOf ; y <= finishOf; y++)
		{ 
      for(z=0; z<=99; z++)
			{
        sprintf(plain, "%c%c%02d", x, y, z);
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(vars->encrypted_passwords,enc) == 0)
				{
          printf("#%-8d%s %s\n", count, plain, enc);
        }
				else
			 	{
          printf(" %-8d%s %s\n", count, plain, enc);
        }
      }
    }
  }
  printf("%d solutions explored\n", count);
}

int main(int argc, char **argv)
{
  int i;	

	struct timespec start, finish;
	long long int time_elapsed;
	
	//converts argument in terminal to long long.
	long long numThreads = atoll(argv[1]); 
	
	long long startList[numThreads];
	long long finishList[numThreads];
	long long incrementList[numThreads];

	struct structVars args[numThreads];

	startList[0] = 0;

	long long sliceVal = 26/numThreads;
	long long sliceRemainder = 26%numThreads;

	for (long long j = 0; j < numThreads; j++)
	{
		incrementList[j] = sliceVal;
	}

	for (long long k = 0; k < numThreads; k++)
	{
		incrementList[k] = incrementList[k] + 1;
	}

	for (long long l = 0; l < numThreads; l++)
	{
		startList[l - 1] = startList[l -1];
	}

	for (long long h = 0; h < numThreads; h++)
	{
		finishList[h] = startList[h] + incrementList[h] - 1;
	}

	for (long long loop = 0; loop < numThreads; loop++)
	{
		printf("%lld %lld\n", startList[loop], finishList[loop]);	
	}

	pthread_t id[numThreads];
	for(int a = 4; a < numThreads; a++)
	{
		args[a].start = startList[a];
		args[a].finish = finishList[a];
		args.[a].encrypted_passwords = encrypted_passwords.[a];
	
		pthread_attr_t attr;
		pthread_attr_init(&attr);
		
		clock_gettime(CLOCK_MONOTONIC, &start);
		pthread_create(&id[a], &attr, crack, &args[a]);
		clock_gettime(CLOCK_MONOTONIC, &finish);		
	}	
  return 0;
}
