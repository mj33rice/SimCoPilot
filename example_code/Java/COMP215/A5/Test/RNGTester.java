import junit.framework.TestCase;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;

/**
 * Interface to a pseudo random number generator that generates
 * numbers between 0.0 and 1.0 and can start over to regenerate
 * exactly the same sequence of values.
 */

interface IPRNG {
  /**
   * Return the next double value between 0.0 and 1.0
   */
  double next();
  
  /**
   * Reset the PRNG to the original seed
   */
  void startOver();
}

/**
 * This is the interface for a vector of double-precision floating point values.
 */
interface IDoubleVector {
  
  /** 
   * Adds another double vector to the contents of this one.  
   * Will throw OutOfBoundsException if the two vectors
   * don't have exactly the same sizes.
   * 
   * @param addThisOne       the double vector to add in
   */
  public void addMyselfToHim (IDoubleVector addToHim) throws OutOfBoundsException;
  
  /**
   * Returns a particular item in the double vector.  Will throw OutOfBoundsException if the index passed
   * in is beyond the end of the vector.
   * 
   * @param whichOne     the index of the item to return
   * @return             the value at the specified index
   */
  public double getItem (int whichOne) throws OutOfBoundsException;

  /**  
   * Add a value to all elements of the vector.
   *
   * @param addMe  the value to be added to all elements
   */
  public void addToAll (double addMe);
  
  /** 
   * Returns a particular item in the double vector, rounded to the nearest integer.  Will throw 
   * OutOfBoundsException if the index passed in is beyond the end of the vector.
   * 
   * @param whichOne    the index of the item to return
   * @return            the value at the specified index
   */
  public long getRoundedItem (int whichOne) throws OutOfBoundsException;
  
  /**
   * This forces the sum of all of the items in the vector to be one, by dividing each item by the
   * total over all items.
   */
  public void normalize ();
  
  /**
   * Sets a particular item in the vector.  
   * Will throw OutOfBoundsException if we are trying to set an item
   * that is past the end of the vector.
   * 
   * @param whichOne     the index of the item to set
   * @param setToMe      the value to set the item to
   */
  public void setItem (int whichOne, double setToMe) throws OutOfBoundsException;

  /**
   * Returns the length of the vector.
   * 
   * @return     the vector length
   */
  public int getLength ();
  
  /**
   * Returns the L1 norm of the vector (this is just the sum of the absolute value of all of the entries)
   * 
   * @return      the L1 norm
   */
  public double l1Norm ();
  
  /**
   * Constructs and returns a new "pretty" String representation of the vector.
   * 
   * @return        the string representation
   */
  public String toString ();
}

/** 
 * Encapsulates the idea of a random object generation algorithm.  The random
 * variable that the algorithm simulates outputs an object of type OutputType.
 * Every concrete class that implements this interface should have a public 
 * constructor of the form:
 * 
 * ConcreteClass (Seed mySeed, ParamType myParams)
 * 
 */
interface IRandomGenerationAlgorithm <OutputType> {
  
  /**
   * Generate another random object
   */
  OutputType getNext ();
  
  /**
   * Resets the sequence of random objects that are created.  The net effect
   * is that if we create the IRandomGenerationAlgorithm object, 
   * then call getNext a bunch of times, and then call startOver (), then 
   * call getNext a bunch of times, we will get exactly the same sequence 
   * of random values the second time around.
   */
  void startOver ();
  
}

/**
 * Wrap the Java random number generator.
 */

class PRNG implements IPRNG {

  /**
   * Java random number generator
   */
  private long seedValue;
  private Random rng;

  /**
   * Build a new pseudo random number generator with the given seed.
   */
  public PRNG(long mySeed) {
    seedValue = mySeed;
    rng = new Random(seedValue);
  }
  
  /**
   * Return the next double value between 0.0 and 1.0
   */
  public double next() {
    return rng.nextDouble();
  }
  
  /**
   * Reset the PRNG to the original seed
   */
  public void startOver() {
    rng.setSeed(seedValue);
  }
}


class Multinomial extends ARandomGenerationAlgorithm <IDoubleVector> {
  
  /**
   * Parameters
   */
  private MultinomialParam params;

  public Multinomial (long mySeed, MultinomialParam myParams) {
    super(mySeed);
    params = myParams;
  }
  
  public Multinomial (IPRNG prng, MultinomialParam myParams) {
    super(prng);
    params = myParams;
  }

  /**
   * Generate another random object
   */
  public IDoubleVector getNext () {
    
    // get an array of doubles; we'll put one random number in each slot
    Double [] myArray = new Double[params.getNumTrials ()];
    for (int i = 0; i < params.getNumTrials (); i++) {
      myArray[i] = genUniform (0, 1.0);  
    }
    
    // now sort them
    Arrays.sort (myArray);
    
    // get the output result
    SparseDoubleVector returnVal = new SparseDoubleVector (params.getProbs ().getLength (), 0.0);
    
    try {
      // now loop through the probs and the observed doubles, and do a merge
      int curPosInProbs = -1;
      double totProbSoFar = 0.0;
      for (int i = 0; i < params.getNumTrials (); i++) {
        while (myArray[i] > totProbSoFar) {
          curPosInProbs++;
          totProbSoFar += params.getProbs ().getItem (curPosInProbs);
        }
        returnVal.setItem (curPosInProbs, returnVal.getItem (curPosInProbs) + 1.0);
      }
      return returnVal;
    } catch (OutOfBoundsException e) {
      System.err.println ("Somehow, within the multinomial sampler, I went out of bounds as I was contructing the output array");
      return null;
    }
  }
  
}

/**
 * This holds a parameterization for the multinomial distribution
 */
class MultinomialParam {
 
  int numTrials;
  IDoubleVector probs;
  
  public MultinomialParam (int numTrialsIn, IDoubleVector probsIn) {
    numTrials = numTrialsIn;
    probs = probsIn;
  }
  
  public int getNumTrials () {
    return numTrials;
  }
  
  public IDoubleVector getProbs () {
    return probs;
  }
}


/*
 * This holds a parameterization for the gamma distribution
 */
class GammaParam {
 
  private double shape, scale, leftmostStep;
  private int numSteps;
  
  public GammaParam (double shapeIn, double scaleIn, double leftmostStepIn, int numStepsIn) {
    shape = shapeIn;
    scale = scaleIn;
    leftmostStep = leftmostStepIn;
    numSteps = numStepsIn;  
  }
  
  public double getShape () {
    return shape;
  }
  
  public double getScale () {
    return scale;
  }
  
  public double getLeftmostStep () {
    return leftmostStep;
  }
  
  public int getNumSteps () {
    return numSteps;
  }
}


class Gamma extends ARandomGenerationAlgorithm <Double> {

  /**
   * Parameters
   */
  private GammaParam params;

  long numShapeOnesToUse;
  private UnitGamma gammaShapeOne;
  private UnitGamma gammaShapeLessThanOne;

  public Gamma(IPRNG prng, GammaParam myParams) {
    super (prng);
    params = myParams;
    setup ();
  }
  
  public Gamma(long mySeed, GammaParam myParams) {
    super (mySeed);
    params = myParams;
    setup ();
  }
  
  private void setup () {
    
    numShapeOnesToUse = (long) Math.floor(params.getShape());
    if (numShapeOnesToUse >= 1) {
      gammaShapeOne = new UnitGamma(getPRNG(), new GammaParam(1.0, 1.0,
                                                        params.getLeftmostStep(), 
                                                        params.getNumSteps()));
    }
    double leftover = params.getShape() - (double) numShapeOnesToUse;
    if (leftover > 0.0) {
      gammaShapeLessThanOne = new UnitGamma(getPRNG(), new GammaParam(leftover, 1.0,
                                                        params.getLeftmostStep(), 
                                                        params.getNumSteps()));
    }
  }

  /**
   * Generate another random object
   */
  public Double getNext () {
    Double value = 0.0;
    for (int i = 0; i < numShapeOnesToUse; i++) {
      value += gammaShapeOne.getNext();
    }
    if (gammaShapeLessThanOne != null)
      value += gammaShapeLessThanOne.getNext ();
    
    return value * params.getScale ();
  }

}

/*
 * This holds a parameterization for the Dirichlet distribution
 */
class DirichletParam {
 
  private IDoubleVector shapes;
  private double leftmostStep;
  private int numSteps;
  
  public DirichletParam (IDoubleVector shapesIn, double leftmostStepIn, int numStepsIn) {
    shapes = shapesIn;
    leftmostStep = leftmostStepIn;
    numSteps = numStepsIn;
  }
  
  public IDoubleVector getShapes () {
    return shapes;
  }
  
  public double getLeftmostStep () {
    return leftmostStep;
  }
  
  public int getNumSteps () {
    return numSteps;
  }
}


class Dirichlet extends ARandomGenerationAlgorithm <IDoubleVector> {

  /**
   * Parameters
   */
  private DirichletParam params;

  public Dirichlet (long mySeed, DirichletParam myParams) {
    super (mySeed);
    params = myParams;
    setup ();
  }
  
  public Dirichlet (IPRNG prng, DirichletParam myParams) {
    super (prng);
    params = myParams;
    setup ();
  }
  
  /**
   * List of gammas that compose the Dirichlet
   */
  private ArrayList<Gamma> gammas = new ArrayList<Gamma>();

  public void setup () {

    IDoubleVector shapes = params.getShapes();
    int i = 0;

    try {
      for (; i<shapes.getLength(); i++) {
        Double d = shapes.getItem(i);
        gammas.add(new Gamma(getPRNG(), new GammaParam(d, 1.0, 
                                                      params.getLeftmostStep(),
                                                      params.getNumSteps())));
      }
    } catch (OutOfBoundsException e) {
      System.err.format("Dirichlet constructor Error: out of bounds access (%d) to shapes\n", i);
      params = null;
      gammas = null;
      return;
    }
  }

  /**
   * Generate another random object
   */
  public IDoubleVector getNext () {
    int length = params.getShapes().getLength();
    IDoubleVector v = new DenseDoubleVector(length, 0.0);

    int i = 0;

    try {
      for (Gamma g : gammas) {
        v.setItem(i++, g.getNext());
      }
    } catch (OutOfBoundsException e) {
      System.err.format("Dirichlet getNext Error: out of bounds access (%d) to result vector\n", i);
      return null;
    }

    v.normalize();

    return v;
  }

}

/**
 * Wrap the Java random number generator.
 */

class ChrisPRNG implements IPRNG {

  /**
   * Java random number generator
   */
  private long seedValue;
  private BigInteger a = new BigInteger ("6364136223846793005");
  private BigInteger c = new BigInteger ("1442695040888963407");
  private BigInteger m = new BigInteger ("2").pow (64);
  private BigInteger X = new BigInteger ("0");
  private double max = Math.pow (2.0, 32);
  
  /**
   * Build a new pseudo random number generator with the given seed.
   */
  public ChrisPRNG(long mySeed) {
    seedValue = mySeed;
    X = new BigInteger (seedValue + "");
  }
  
  /**
   * Return the next double value between 0.0 and 1.0
   */
  public double next() {
    X = X.multiply (a).add (c).mod (m);
    return X.shiftRight (32).intValue () / max + 0.5;
  }
  
  /**
   * Reset the PRNG to the original seed
   */
  public void startOver() {
    X = new BigInteger (seedValue + "");
  }
}

class Main {

	public static void main (String [] args) {
		IPRNG foo = new ChrisPRNG (0);
		for (int i = 0; i < 10; i++) {
			System.out.println (foo.next ());
		}
	}
}

/** 
 * Encapsulates the idea of a random object generation algorithm.  The random
 * variable that the algorithm simulates is parameterized using an object of
 * type InputType.  Every concrete class that implements this interface should
 * have a constructor of the form:
 * 
 * ConcreteClass (Seed mySeed, ParamType myParams)
 * 
 */
abstract class ARandomGenerationAlgorithm <OutputType> implements IRandomGenerationAlgorithm <OutputType> {

  /**
   * There are two ways that this guy gets random numbers.  Either he gets them from a "parent" (this is
   * the ARandomGenerationAlgorithm object who created him) or he manufctures them himself if he has been
   * created by someone other than another ARandomGenerationAlgorithm object.
   */
  
  private IPRNG prng;
    
  // this constructor is called when we want an ARandomGenerationAlgorithm who uses his own rng
  protected ARandomGenerationAlgorithm(long mySeed) {
    prng = new ChrisPRNG(mySeed);  
  }
  
  // this one is called when we want a ARandomGenerationAlgorithm who uses someone elses rng
  protected ARandomGenerationAlgorithm(IPRNG useMe) {
    prng = useMe;
  }

  protected IPRNG getPRNG() {
    return prng;
  }
  
  /**
   * Generate another random object
   */
  abstract public OutputType getNext ();
  
  /**
   * Resets the sequence of random objects that are created.  The net effect
   * is that if we create the IRandomGenerationAlgorithm object, 
   * then call getNext a bunch of times, and then call startOver (), then 
   * call getNext a bunch of times, we will get exactly the same sequence 
   * of random values the second time around.
   */
  public void startOver () {
    prng.startOver();
  }

  /**
   * Generate a random number uniformly between low and high
   */
  protected double genUniform(double low, double high) {
    double r = prng.next();
    r *= high - low;
    r += low;
    return r;
  }

}


class Uniform extends ARandomGenerationAlgorithm <Double> {
  
  /**
   * Parameters
   */
  private UniformParam params;

  public Uniform(long mySeed, UniformParam myParams) {
    super(mySeed);
    params = myParams;
  }
  
  public Uniform(IPRNG prng, UniformParam myParams) {
    super(prng);
    params = myParams;
  }

  /**
   * Generate another random object
   */
  public Double getNext () {
    return genUniform(params.getLow(), params.getHigh());
  }
  
}

/**
 * This holds a parameterization for the uniform distribution
 */
class UniformParam {
 
  double low, high;
  
  public UniformParam (double lowIn, double highIn) {
    low = lowIn;
    high = highIn;
  }
  
  public double getLow () {
    return low;
  }
  
  public double getHigh () {
    return high;
  }
}

class UnitGamma extends ARandomGenerationAlgorithm <Double> {

  /**
   * Parameters
   */
  private GammaParam params;

  private class RejectionSampler {
    private UniformParam xRegion;
    private UniformParam yRegion;
    private double area;

    protected RejectionSampler(double xmin, double xmax, 
                               double ymin, double ymax) {   
      xRegion = new UniformParam(xmin, xmax);
      yRegion = new UniformParam(ymin, ymax);
      area = (xmax - xmin) * (ymax - ymin);
    }

    protected Double tryNext() {
      Double x = genUniform (xRegion.getLow (), xRegion.getHigh ());
      Double y = genUniform (yRegion.getLow (), yRegion.getHigh ());
      
      if (y <= fofx(x)) {
          return x;
      } else {
          return null;
      }
    }

    protected double getArea() {
      return area;
    }
  }
  
  /**
   * Uniform generators
   */
  // Samplers for each step
  private ArrayList<RejectionSampler> samplers = new ArrayList<RejectionSampler>();

  // Total area of all envelopes
  private double totalArea;

  private double fofx(double x) {
    double k = params.getShape();
    return Math.pow(x, k-1) * Math.exp(-x);
  }

  private void setup () {
    if (params.getShape() > 1.0 || params.getScale () > 1.0000000001 || params.getScale () < .999999999 ) {
      System.err.println("UnitGamma must have shape <= 1.0 and a scale of one");
      params = null;
      samplers = null;
      return;
    }

    totalArea = 0.0;
    

    for (int i=0; i<params.getNumSteps(); i++) {
        double left = params.getLeftmostStep() * Math.pow (2.0, i);
        double fx = fofx(left);
        samplers.add(new RejectionSampler(left, left*2.0, 0, fx));
        totalArea += (left*2.0 - left) * fx;
    }
  }
 
  public UnitGamma(IPRNG prng, GammaParam myParams) {
    super(prng);
    params = myParams;
    setup ();
  } 
  
  public UnitGamma(long mySeed, GammaParam myParams) {
    super(mySeed);
    params = myParams;
    setup ();
  }

  /**
   * Generate another random object
   */
  public Double getNext () {
    while (true) {
      double step = genUniform(0.0, totalArea);
      double area = 0.0;
      for (RejectionSampler s : samplers) {
        area += s.getArea();
        if (area >= step) {
          // Found the right sampler
          Double x = s.tryNext();
          if (x != null) {
            return x;
          }
          break;
        }
      }
    }
  }
}


/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
public class RNGTester extends TestCase {
  
  
  // checks to see if two values are close enough
  private void checkCloseEnough (double observed, double goal, double tolerance, String whatCheck, String dist) { 
    
    // first compute the allowed error 
    double allowedError = goal * tolerance; 
    if (allowedError < tolerance) 
      allowedError = tolerance; 
    
    // print the result
    System.out.println ("Got " + observed + ", expected " + goal + " when I was checking the " + whatCheck + " of the " + dist);
    
    // if it is too large, then fail 
    if (!(observed > goal - allowedError && observed < goal + allowedError)) 
      fail ("Got " + observed + ", expected " + goal + " when I was checking the " + whatCheck + " of the " + dist); 
  } 
 
  /**
   * This routine checks whether the observed mean for a one-dim RNG is close enough to the expected mean.
   * Note the weird "runTest" parameter.  If this is set to true, the test for correctness is actually run.
   * If it is set to false, the test is not run, and only the observed mean is returned to the caller.
   * This is done so that we can "turn off" the correctness check, when we are only using this routine 
   * to compute an observed mean in order to check if the seeding is working.  This way, we don't check
   * too many things in one single test.
   */
  private double checkMean (IRandomGenerationAlgorithm<Double> myRNG, double expectedMean, boolean runTest, String dist) {
  
    int numTrials = 100000;
    double total = 0.0;
    for (int i = 0; i < numTrials; i++) {
      total += myRNG.getNext ();  
    }
    
    if (runTest)
      checkCloseEnough (total / numTrials, expectedMean, 10e-3, "mean", dist);
    
    return total / numTrials;
  }
  
  /**
   * This checks whether the standard deviation for a one-dim RNG is close enough to the expected std dev.
   */
  private void checkStdDev (IRandomGenerationAlgorithm<Double> myRNG, double expectedVariance, String dist) {
  
    int numTrials = 100000;
    double total = 0.0;
    double totalSquares = 0.0;
    for (int i = 0; i < numTrials; i++) {
      double next = myRNG.getNext ();
      total += next;
      totalSquares += next * next;
    }
    
    double observedVar = -(total / numTrials) * (total / numTrials) + totalSquares / numTrials;
    
    checkCloseEnough (Math.sqrt(observedVar), Math.sqrt(expectedVariance), 10e-3, "standard deviation", dist);
  }
  
  /**
   * Tests whether the startOver routine works correctly.  To do this, it generates a sequence of random
   * numbers, and computes the mean of them.  Then it calls startOver, and generates the mean once again.
   * If the means are very, very close to one another, then the test case passes.
   */
  public void testUniformReset () {
    
    double low = 0.5;
    double high = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG = new Uniform (745664, new UniformParam (low, high));
    double result1 = checkMean (myRNG, 0, false, "");
    myRNG.startOver ();
    double result2 = checkMean (myRNG, 0, false, "");
    assertTrue ("Failed check for uniform reset capability", Math.abs (result1 - result2) < 10e-10);
  }
  
  /** 
   * Tests whether seeding is correctly used.  This is run just like the startOver test above, except
   * that here we create two sequences using two different seeds, and then we verify that the observed
   * means were different from one another.
   */
  public void testUniformSeeding () {
    
    double low = 0.5;
    double high = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG1 = new Uniform (745664, new UniformParam (low, high));
    IRandomGenerationAlgorithm<Double> myRNG2 = new Uniform (2334, new UniformParam (low, high));
    double result1 = checkMean (myRNG1, 0, false, "");
    double result2 = checkMean (myRNG2, 0, false, "");
    assertTrue ("Failed check for uniform seeding correctness", Math.abs (result1 - result2) > 10e-10);
  }
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testUniform1 () {
    
    double low = 0.5;
    double high = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG = new Uniform (745664, new UniformParam (low, high));
    checkMean (myRNG, low / 2.0 + high / 2.0, true, "Uniform (" + low + ", " + high + ")");
    checkStdDev (myRNG, (high - low) * (high - low) / 12.0, "Uniform (" + low + ", " + high + ")");
  }
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testUniform2 () {

    double low = -123456.0;
    double high = 233243.0;
    IRandomGenerationAlgorithm<Double> myRNG = new Uniform (745664, new UniformParam (low, high));
    checkMean (myRNG, low / 2.0 + high / 2.0, true, "Uniform (" + low + ", " + high + ")");
    checkStdDev (myRNG, (high - low) * (high - low) / 12.0, "Uniform (" + low + ", " + high + ")");
  }
  
  /**
   * Tests whether the startOver routine works correctly.  To do this, it generates a sequence of random
   * numbers, and computes the mean of them.  Then it calls startOver, and generates the mean once again.
   * If the means are very, very close to one another, then the test case passes.
   */  
  public void testUnitGammaReset () {
    
    double shape = 0.5;
    double scale = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG = new UnitGamma (745664, new GammaParam (shape, scale, 10e-40, 150));
    double result1 = checkMean (myRNG, 0, false, "");
    myRNG.startOver ();
    double result2 = checkMean (myRNG, 0, false, "");
    assertTrue ("Failed check for unit gamma reset capability", Math.abs (result1 - result2) < 10e-10);
  }
  /** 
   * Tests whether seeding is correctly used.  This is run just like the startOver test above, except
   * that here we create two sequences using two different seeds, and then we verify that the observed
   * means were different from one another.
   */  
  public void testUnitGammaSeeding () {
    
    double shape = 0.5;
    double scale = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG1 = new UnitGamma (745664, new GammaParam (shape, scale, 10e-40, 150));
    IRandomGenerationAlgorithm<Double> myRNG2 = new UnitGamma (232, new GammaParam (shape, scale, 10e-40, 150));
    double result1 = checkMean (myRNG1, 0, false, "");
    double result2 = checkMean (myRNG2, 0, false, "");
    assertTrue ("Failed check for unit gamma seeding correctness", Math.abs (result1 - result2) > 10e-10);
  }
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testUnitGamma1 () {
    
    double shape = 0.5;
    double scale = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG = new UnitGamma (745664, new GammaParam (shape, scale, 10e-40, 150));
    checkMean (myRNG, shape * scale, true, "Gamma (" + shape + ", " + scale + ")");
    checkStdDev (myRNG, shape * scale * scale, "Gamma (" + shape + ", " + scale + ")");
  }

  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testUnitGamma2 () {
    
    double shape = 0.05;
    double scale = 1.0;
    IRandomGenerationAlgorithm<Double> myRNG = new UnitGamma (6755, new GammaParam (shape, scale, 10e-40, 150));
    checkMean (myRNG, shape * scale, true, "Gamma (" + shape + ", " + scale + ")");
    checkStdDev (myRNG, shape * scale * scale, "Gamma (" + shape + ", " + scale + ")");
  }
  
  /**
   * Tests whether the startOver routine works correctly.  To do this, it generates a sequence of random
   * numbers, and computes the mean of them.  Then it calls startOver, and generates the mean once again.
   * If the means are very, very close to one another, then the test case passes.
   */  
  public void testGammaReset () {
    
    double shape = 15.09;
    double scale = 3.53;
    IRandomGenerationAlgorithm<Double> myRNG = new Gamma (27, new GammaParam (shape, scale, 10e-40, 150));
    double result1 = checkMean (myRNG, 0, false, "");
    myRNG.startOver ();
    double result2 = checkMean (myRNG, 0, false, "");
    assertTrue ("Failed check for gamma reset capability", Math.abs (result1 - result2) < 10e-10);
  }
  
  /** 
   * Tests whether seeding is correctly used.  This is run just like the startOver test above, except
   * that here we create two sequences using two different seeds, and then we verify that the observed
   * means were different from one another.
   */  
  public void testGammaSeeding () {
    
    double shape = 15.09;
    double scale = 3.53;
    IRandomGenerationAlgorithm<Double> myRNG1 = new Gamma (27, new GammaParam (shape, scale, 10e-40, 150));
    IRandomGenerationAlgorithm<Double> myRNG2 = new Gamma (297, new GammaParam (shape, scale, 10e-40, 150));
    double result1 = checkMean (myRNG1, 0, false, "");
    double result2 = checkMean (myRNG2, 0, false, "");
    assertTrue ("Failed check for gamma seeding correctness", Math.abs (result1 - result2) > 10e-10);
  }

  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testGamma1 () {
    
    double shape = 5.88;
    double scale = 34.0;
    IRandomGenerationAlgorithm<Double> myRNG = new Gamma (27, new GammaParam (shape, scale, 10e-40, 150));
    checkMean (myRNG, shape * scale, true, "Gamma (" + shape + ", " + scale + ")");
    checkStdDev (myRNG, shape * scale * scale, "Gamma (" + shape + ", " + scale + ")");
  }
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testGamma2 () {
    
    double shape = 15.09;
    double scale = 3.53;
    IRandomGenerationAlgorithm<Double> myRNG = new Gamma (27, new GammaParam (shape, scale, 10e-40, 150));
    checkMean (myRNG, shape * scale, true, "Gamma (" + shape + ", " + scale + ")");
    checkStdDev (myRNG, shape * scale * scale, "Gamma (" + shape + ", " + scale + ")");
  }
  
  /**
   * This checks the sub of the absolute differences between the entries of two vectors; if it is too large, then
   * a jUnit error is generated
   */
  public void checkTotalDiff (IDoubleVector actual, IDoubleVector expected, double error, String test, String dist) throws OutOfBoundsException {
    double totError = 0.0;
    for (int i = 0; i < actual.getLength (); i++) {
      totError += Math.abs (actual.getItem (i) - expected.getItem (i));
    }
    checkCloseEnough (totError, 0.0, error, test, dist);
  }
  
  /**
   * Computes the difference between the observed mean of a multi-dim random variable, and the expected mean.
   * The difference is returned as the sum of the parwise absolute values in the two vectors.  This is used
   * for the seeding and startOver tests for the two multi-dim random variables.
   */
  public double computeDiffFromMean (IRandomGenerationAlgorithm<IDoubleVector> myRNG, IDoubleVector expectedMean, int numTrials) {
    
    // set up the total so far
    try {
      IDoubleVector firstOne = myRNG.getNext ();
      DenseDoubleVector meanObs = new DenseDoubleVector (firstOne.getLength (), 0.0);
    
      // add in a bunch more
      for (int i = 0; i < numTrials; i++) {
        IDoubleVector next = myRNG.getNext ();
        next.addMyselfToHim (meanObs);
      }
    
      // compute the total difference from the mean
      double returnVal = 0;
      for (int i = 0; i < meanObs.getLength (); i++) {
        returnVal += Math.abs (meanObs.getItem (i) / numTrials - expectedMean.getItem (i));
      }
      
      // and return it
      return returnVal;
      
    } catch (OutOfBoundsException e) {
      fail ("I got an OutOfBoundsException when running getMean... bad vector length back?");  
      return 0.0;
    }
  }
                                     
  /**
   * Checks the observed mean and variance for a multi-dim random variable versus the theoretically expected values
   * for these quantities.  If the observed differs substantially from the expected, the test case is failed.  This
   * is used for checking the correctness of the multi-dim random variables.
   */
  public void checkMeanAndVar (IRandomGenerationAlgorithm<IDoubleVector> myRNG, 
      IDoubleVector expectedMean, IDoubleVector expectedStdDev, double errorMean, double errorStdDev, int numTrials, String dist) {
  
    // set up the total so far
    try {
      IDoubleVector firstOne = myRNG.getNext ();
      DenseDoubleVector meanObs = new DenseDoubleVector (firstOne.getLength (), 0.0);
      DenseDoubleVector stdDevObs = new DenseDoubleVector (firstOne.getLength (), 0.0);
    
      // add in a bunch more
      for (int i = 0; i < numTrials; i++) {
        IDoubleVector next = myRNG.getNext ();
        next.addMyselfToHim (meanObs);
        for (int j = 0; j < next.getLength (); j++) {
          stdDevObs.setItem (j, stdDevObs.getItem (j) + next.getItem (j) * next.getItem (j));  
        }
      }
    
      // divide by the number of trials to get the mean
      for (int i = 0; i < meanObs.getLength (); i++) {
        meanObs.setItem (i, meanObs.getItem (i) / numTrials);
        stdDevObs.setItem (i, Math.sqrt (stdDevObs.getItem (i) / numTrials - meanObs.getItem (i) * meanObs.getItem (i)));
      }
    
      // see if the mean and var are acceptable
      checkTotalDiff (meanObs, expectedMean, errorMean, "total distance from true mean", dist);
      checkTotalDiff (stdDevObs, expectedStdDev, errorStdDev, "total distance from true standard deviation", dist);
      
    } catch (OutOfBoundsException e) {
      fail ("I got an OutOfBoundsException when running getMean... bad vector length back?");  
    }
  }
  
  /**
   * Tests whether the startOver routine works correctly.  To do this, it generates a sequence of random
   * numbers, and computes the mean of them.  Then it calls startOver, and generates the mean once again.
   * If the means are very, very close to one another, then the test case passes.
   */  
  public void testMultinomialReset () {
    
    try {
      // first set up a vector of probabilities
      int len = 100;
      SparseDoubleVector probs = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i += 2) {
        probs.setItem (i, i); 
      }
      probs.normalize ();
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Multinomial (27, new MultinomialParam (1024, probs));
    
      // and check the mean...
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, probs.getItem (i) * 1024);
      }
      
      double res1 = computeDiffFromMean (myRNG, expectedMean, 500);
      myRNG.startOver ();
      double res2 = computeDiffFromMean (myRNG, expectedMean, 500);
      
      assertTrue ("Failed check for multinomial reset", Math.abs (res1 - res2) < 10e-10);
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the multinomial."); 
    }
   
  }

  /** 
   * Tests whether seeding is correctly used.  This is run just like the startOver test above, except
   * that here we create two sequences using two different seeds, and then we verify that the observed
   * means were different from one another.
   */  
  public void testMultinomialSeeding () {
    
    try {
      // first set up a vector of probabilities
      int len = 100;
      SparseDoubleVector probs = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i += 2) {
        probs.setItem (i, i); 
      }
      probs.normalize ();
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG1 = new Multinomial (27, new MultinomialParam (1024, probs));
      IRandomGenerationAlgorithm<IDoubleVector> myRNG2 = new Multinomial (2777, new MultinomialParam (1024, probs));
      
      // and check the mean...
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, probs.getItem (i) * 1024);
      }
      
      double res1 = computeDiffFromMean (myRNG1, expectedMean, 500);
      double res2 = computeDiffFromMean (myRNG2, expectedMean, 500);
      
      assertTrue ("Failed check for multinomial seeding", Math.abs (res1 - res2) > 10e-10);
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the multinomial."); 
    }
   
  }
  
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */  
  public void testMultinomial1 () {
    
    try {
      // first set up a vector of probabilities
      int len = 100;
      SparseDoubleVector probs = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i += 2) {
        probs.setItem (i, i); 
      }
      probs.normalize ();
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Multinomial (27, new MultinomialParam (1024, probs));
    
      // and check the mean... we repeatedly double the prob vector to multiply it by 1024
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      DenseDoubleVector expectedStdDev = new DenseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, probs.getItem (i) * 1024);
        expectedStdDev.setItem (i, Math.sqrt (probs.getItem (i) * 1024 * (1.0 - probs.getItem (i))));
      }
      
      checkMeanAndVar (myRNG, expectedMean, expectedStdDev, 5.0, 5.0, 5000, "multinomial number one");
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the multinomial."); 
    }
   
  }

  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */  
  public void testMultinomial2 () {
    
    try {
      // first set up a vector of probabilities
      int len = 1000;
      SparseDoubleVector probs = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len - 1; i ++) {
        probs.setItem (i, 0.0001); 
      }
      probs.setItem (len - 1, 100);
      probs.normalize ();
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Multinomial (27, new MultinomialParam (10, probs));
    
      // and check the mean... we repeatedly double the prob vector to multiply it by 1024
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      DenseDoubleVector expectedStdDev = new DenseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, probs.getItem (i) * 10);
        expectedStdDev.setItem (i, Math.sqrt (probs.getItem (i) * 10 * (1.0 - probs.getItem (i))));
      }
      
      checkMeanAndVar (myRNG, expectedMean, expectedStdDev, 0.05, 5.0, 5000, "multinomial number two");
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the multinomial."); 
    }
   
  }
  
  /**
   * Tests whether the startOver routine works correctly.  To do this, it generates a sequence of random
   * numbers, and computes the mean of them.  Then it calls startOver, and generates the mean once again.
   * If the means are very, very close to one another, then the test case passes.
   */  
  public void testDirichletReset () {
    
    try {
      
      // first set up a vector of shapes
      int len = 100;
      SparseDoubleVector shapes = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        shapes.setItem (i, 0.05); 
      }
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Dirichlet (27, new DirichletParam (shapes, 10e-40, 150));
    
      // compute the expected mean
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      double norm = shapes.l1Norm ();
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, shapes.getItem (i) / norm);
      }
      
      double res1 = computeDiffFromMean (myRNG, expectedMean, 500);
      myRNG.startOver ();
      double res2 = computeDiffFromMean (myRNG, expectedMean, 500);
      
      assertTrue ("Failed check for Dirichlet reset", Math.abs (res1 - res2) < 10e-10);
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the Dirichlet."); 
    }
   
  }

  /** 
   * Tests whether seeding is correctly used.  This is run just like the startOver test above, except
   * that here we create two sequences using two different seeds, and then we verify that the observed
   * means were different from one another.
   */  
  public void testDirichletSeeding () {
    
    try {
      
      // first set up a vector of shapes
      int len = 100;
      SparseDoubleVector shapes = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        shapes.setItem (i, 0.05); 
      }
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG1 = new Dirichlet (27, new DirichletParam (shapes, 10e-40, 150));
      IRandomGenerationAlgorithm<IDoubleVector> myRNG2 = new Dirichlet (92, new DirichletParam (shapes, 10e-40, 150));
    
      // compute the expected mean
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      double norm = shapes.l1Norm ();
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, shapes.getItem (i) / norm);
      }
      
      double res1 = computeDiffFromMean (myRNG1, expectedMean, 500);
      double res2 = computeDiffFromMean (myRNG2, expectedMean, 500);
      
      assertTrue ("Failed check for Dirichlet seeding", Math.abs (res1 - res2) > 10e-10);
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the Dirichlet."); 
    }
   
  }

  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testDirichlet1 () {
    
    try {
      // first set up a vector of shapes
      int len = 100;
      SparseDoubleVector shapes = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len; i++) {
        shapes.setItem (i, 0.05); 
      }
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Dirichlet (27, new DirichletParam (shapes, 10e-40, 150));
    
      // compute the expected mean and var
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      DenseDoubleVector expectedStdDev = new DenseDoubleVector (len, 0.0);
      double norm = shapes.l1Norm ();
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, shapes.getItem (i) / norm);
        expectedStdDev.setItem (i, Math.sqrt (shapes.getItem (i) * (norm - shapes.getItem (i)) / 
                                   (norm * norm * (1.0 + norm))));
      }
      
      checkMeanAndVar (myRNG, expectedMean, expectedStdDev, 0.1, 0.3, 5000, "Dirichlet number one");
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the Dirichlet."); 
    }
   
  }
  
  /**
   * Generates a bunch of random variables, and then uses the know formulas for the mean and variance of those
   * variables to verify that the observed mean and variance are reasonable; if they are, this is a strong 
   * indication that the variables are being generated correctly
   */
  public void testDirichlet2 () {
    
    try {
      // first set up a vector of shapes
      int len = 300;
      SparseDoubleVector shapes = new SparseDoubleVector (len, 0.0);
      for (int i = 0; i < len - 10; i++) {
        shapes.setItem (i, 0.05); 
      }
      for (int i = len - 9; i < len; i++) {
        shapes.setItem (i, 100.0); 
      }
    
      // now, set up a distribution
      IRandomGenerationAlgorithm<IDoubleVector> myRNG = new Dirichlet (27, new DirichletParam (shapes, 10e-40, 150));
    
      // compute the expected mean and var
      DenseDoubleVector expectedMean = new DenseDoubleVector (len, 0.0);
      DenseDoubleVector expectedStdDev = new DenseDoubleVector (len, 0.0);
      double norm = shapes.l1Norm ();
      for (int i = 0; i < len; i++) {
        expectedMean.setItem (i, shapes.getItem (i) / norm);
        expectedStdDev.setItem (i, Math.sqrt (shapes.getItem (i) * (norm - shapes.getItem (i)) / 
                                   (norm * norm * (1.0 + norm))));
      }
      
      checkMeanAndVar (myRNG, expectedMean, expectedStdDev, 0.01, 0.01, 5000, "Dirichlet number one");
      
    } catch (Exception e) {
      fail ("Got some sort of exception when I was testing the Dirichlet."); 
    }
   
  }
  
}
