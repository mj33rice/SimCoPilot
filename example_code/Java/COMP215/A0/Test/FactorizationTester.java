import junit.framework.TestCase;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.Math;

/**
 * This class implements a relatively simple algorithm for computing
 * (and printing) the prime factors of a number.  At initialization,
 * a list of primes is computed.  Given a number, this list is then
 * used to efficiently print the prime factors of the number.
 */
class PrimeFactorizer {

  // this class will store all primes between 2 and upperBound
  private int upperBound;
  
  // this class will be able to factorize all primes between 1 and maxNumberToFactorize
  private long maxNumberToFactorize;
  
  // this is a list of all of the prime numbers between 2 and upperBound
  private int [] allPrimes;
  
  // this is the number of primes found between 2 and upperBound
  private int numPrimes;
  
  // this is the output stream that the factorization will be printed to
  private PrintStream resultStream;
  
  /**
   * This prints the prime factorization of a number in a "pretty" fashion.
   * If the number passed in exceeds "upperBound", then a nice error message
   * is printed.  
   */
  public void printPrimeFactorization (long numToFactorize) {
    
    // If we are given a number that's too small/big to factor, tell the user
    if (numToFactorize < 1) {
      resultStream.format ("Can't factorize a number less than 1");
      return;
    }
    if (numToFactorize > maxNumberToFactorize) {
      resultStream.format ("%d is too large to factorize", numToFactorize);
      return;
    }
    
    // Now get ready to print the factorization
    resultStream.format ("Prime factorization of %d is: ", numToFactorize);
   
    // our basic tactice will be to find all of the primes that divide evenly into
    // our target.  When we find one, print it out and reduce the target accrordingly.
    // When the target gets down to one, we are done.
    
    // this is the index of the next prime we need to try
    int curPosInAllPrimes = 0;
    
    // tells us we are still waiting to print the first prime... important so we
    // don't print the "x" symbol before the first one
    boolean firstOne = true;
    
    while (numToFactorize > 1 && curPosInAllPrimes < numPrimes) {
    
      // if the current prime divides evenly into the target, we've got a prime factor
      if (numToFactorize % allPrimes[curPosInAllPrimes] == 0) {
        
        // if it's the first one, we don't need to print a "x"
        // print the factor & reset the firstOne flag
        if (firstOne) {
          resultStream.format ("%d", allPrimes[curPosInAllPrimes]);
          firstOne = false;
          
        // otherwise, print the factor pre-pended with an "x"
        } else {
          resultStream.format (" x %d", allPrimes[curPosInAllPrimes]);
        }
        
        // remove that prime factor from the target
        numToFactorize /= allPrimes[curPosInAllPrimes];
        
      // if the current prime does not divde evenly, try the next one
      } else {
        curPosInAllPrimes++;
      }
    }
    
    // if we never printed any factors, then display the number itself
    if (firstOne) {
      resultStream.format ("%d", numToFactorize);
    // Otherwsie print the factors connected by 'x'
    } else if (numToFactorize > 1) {
      resultStream.format (" x %d", numToFactorize);
    }
  }
  
  
  /**
   * This is the constructor.  What it does is to fill the array allPrimes with all
   * of the prime numbers from 2 through sqrt(maxNumberToFactorize).  This array of primes
   * can then subsequently be used by the printPrimeFactorization method to actually
   * compute the prime factorization of a number.  The method also sets upperBound
   * (the largest number we checked for primality) and numPrimes (the number of primes
   * in allPimes).  The algorithm used to compute the primes is the classic "sieve of 
   * Eratosthenes" algorithm.
   */
  public PrimeFactorizer (long maxNumberToFactorizeIn, PrintStream outputStream) {
    
    resultStream = outputStream;
    maxNumberToFactorize = maxNumberToFactorizeIn;
    
    // initialize the two class variables- 'upperBound' and 'allPrimes' note that the prime candidates are
    // from 2 to upperBound, so there are upperBound prime candidates intially
    upperBound = (int) Math.ceil (Math.sqrt (maxNumberToFactorize));
    allPrimes = new int[upperBound - 1];
    
    // numIntsInList is the number of ints currently in the list.
    int numIntsInList = upperBound - 1;
    
    // the number of primes so far is zero
    numPrimes = 0;
    
    // write all of the candidate numbers to the list... this is all of the numbers
    // from two to upperBound
    for (int i = 0; i < upperBound - 1; i++) {
      allPrimes[i] = i + 2;
    }
    
    // now we keep removing numbers from the list until we have only primes
    while (numIntsInList > numPrimes) {
      
      // curPos tells us the last slot that has a "good" (still possibly prime) value.
      int curPos = numPrimes + 1;
     
      // the front of the list is a prime... kill everyone who is a multiple of it
      for (int i = numPrimes + 1; i < numIntsInList; i++) {
        
        // if the dude at position i is not a multiple of the current prime, then
        // we keep him; otherwise, he'll be lost
        if (allPrimes[i] % allPrimes[numPrimes] != 0) {
          allPrimes[curPos] = allPrimes[i];
          curPos++;
        }
      }
      
      // the number of ints in the list is now equal to the last slot we wrote a value to
      numIntsInList = curPos;
      
      // and the guy at the front of the list is now considered a prime
      numPrimes++;
      
    }
  }
  

}

/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
 
public class FactorizationTester extends TestCase {
    /**
   * Stream to hold the result of each test.
   */
  private ByteArrayOutputStream result;

  /**
   * Wrapper to provide printing functionality.
   */
  private PrintStream outputStream;

  /**
   * Setup the output streams before each test.
   */   
  protected void setUp () {
    result = new ByteArrayOutputStream();
    outputStream = new PrintStream(result);
  }    
  
  /**
   * Print a nice message about each test.
   */
  private void printInfo(String description, String expected, String actual) {
    System.out.format ("\n%s\n", description);
    System.out.format ("output:\t%s\n", actual);
    System.out.format ("correct:\t%s\n", expected);
  }

  /**
   * These are the various test methods.  The first 10 test factorizations
   * using the object constructed to factorize numbers up to 100; the latter
   * 7 use the object to factorize numbers up to 100000.
   */
  public void test100MaxFactorize1() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (1);

    // Print Results
    String expected = new String("Prime factorization of 1 is: 1");
    printInfo("Factorizing 1 using max 100", expected, result.toString());

    // Check if test passed by comparing the expected and actual results
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 7 
  public void test100MaxFactorize7() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (7);

    // Print Results
    String expected = new String("Prime factorization of 7 is: 7");
    printInfo("Factorizing 7 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 5
  public void test100MaxFactorize5() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (5);

    // Print Results
    String expected = new String("Prime factorization of 5 is: 5");
    printInfo("Factorizing 5 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
   
  // Test the factorization of 30
  public void test100MaxFactorize30() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (30);

    // Print Results
    String expected = new String("Prime factorization of 30 is: 2 x 3 x 5");
    printInfo("Factorizing 30 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 81
  public void test100MaxFactorize81() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (81);

    // Print Results
    String expected = new String("Prime factorization of 81 is: 3 x 3 x 3 x 3");
    printInfo("Factorizing 81 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 71
  public void test100MaxFactorize71() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (71);

    // Print Results
    String expected = new String("Prime factorization of 71 is: 71");
    printInfo("Factorizing 71 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 100
  public void test100MaxFactorize100() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (100);

    // Print Results
    String expected = new String("Prime factorization of 100 is: 2 x 2 x 5 x 5");
    printInfo("Factorizing 100 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 101
  public void test100MaxFactorize101() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (101);

    // Print Results
    String expected = new String("101 is too large to factorize");
    printInfo("Factorizing 101 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 0
  public void test100MaxFactorize0() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (0);

    // Print Results
    String expected = new String("Can't factorize a number less than 1");
    printInfo("Factorizing 0 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  // Test the factorization of 97
  public void test100MaxFactorize97() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100 = new PrimeFactorizer(100, outputStream);
    myFactorizerMax100.printPrimeFactorization (97);

    // Print Results
    String expected = new String("Prime factorization of 97 is: 97");
    printInfo("Factorizing 97 using max 100", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 34534
  // factorize numbers up to 100000000
  public void test100000000MaxFactorize34534() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
    myFactorizerMax100000000.printPrimeFactorization (34534);

    // Print Results
    String expected = new String("Prime factorization of 34534 is: 2 x 31 x 557");
    printInfo("Factorizing 34534 using max 100000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 4339
  // factorize numbers up to 100000000
  public void test100000000MaxFactorize4339() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
    myFactorizerMax100000000.printPrimeFactorization (4339);

    // Print Results
    String expected = new String("Prime factorization of 4339 is: 4339");
    printInfo("Factorizing 4339 using max 100000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 65536
  // factorize numbers up to 100000000
  public void test100000000MaxFactorize65536() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
    myFactorizerMax100000000.printPrimeFactorization (65536);

    // Print Results
    String expected = new String("Prime factorization of 65536 is: 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2 x 2");
    printInfo("Factorizing 65536 using max 100000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 99797
  // factorize numbers up to 100000000
  public void test100000000MaxFactorize99797() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
    myFactorizerMax100000000.printPrimeFactorization (99797);

    // Print Results
    String expected = new String("Prime factorization of 99797 is: 23 x 4339");
    printInfo("Factorizing 99797 using max 100000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  // Test the factorization of 307
  // factorize numbers up to 100000000
  public void test100000000MaxFactorize307() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000 = new PrimeFactorizer(100000000, outputStream);
    myFactorizerMax100000000.printPrimeFactorization (307);

    // Print Results
    String expected = new String("Prime factorization of 307 is: 307");
    printInfo("Factorizing 307 using max 100000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }

  // Test the factorization of 38845248344
  // factorize numbers up to 100000000000
  public void test100000000000MaxFactorize38845248344() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
    myFactorizerMax100000000000.printPrimeFactorization (38845248344L);

    // Print Results
    String expected = new String("Prime factorization of 38845248344 is: 2 x 2 x 2 x 7 x 693665149");
    printInfo("Factorizing 38845248344 using max 100000000000", expected, result.toString());
    
    // Check if test passed
    assertEquals (expected, result.toString());
    
    result.reset ();
    myFactorizerMax100000000000.printPrimeFactorization (24210833220L);
    expected = new String("Prime factorization of 24210833220 is: 2 x 2 x 3 x 3 x 5 x 7 x 17 x 19 x 19 x 31 x 101");
    printInfo("Factorizing 24210833220 using max 100000000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  public void test100000000000MaxFactorize5387457483444() {
    // Factorize the number
    PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
    myFactorizerMax100000000000.printPrimeFactorization (5387457483444L);

    // Print Results
    String expected = new String("5387457483444 is too large to factorize");
    printInfo("Factorizing 5387457483444 using max 100000000000", expected, result.toString());

    // Check if test passed
    assertEquals (expected, result.toString());
  }
  
  public void testSpeed() {
    
    // here we factorize 10,000 large numbers; make sure that the constructor correctly sets
    // up the array of primes just once... if it does not, it'll take too long to run
    PrimeFactorizer myFactorizerMax100000000000 = new PrimeFactorizer(100000000000L, outputStream);
    for (long i = 38845248344L; i < 38845248344L + 50000; i++) {
      myFactorizerMax100000000000.printPrimeFactorization (i);
      if (i == 38845248344L) {
        // Compare the Prime factorization of 38845248344 to the expected result
        String expected = new String("Prime factorization of 38845248344 is: 2 x 2 x 2 x 7 x 693665149"); 
        assertEquals (expected, result.toString());
      }
    }
    
    // if we make it here, we pass the test
    assertTrue (true);
  }
  
}