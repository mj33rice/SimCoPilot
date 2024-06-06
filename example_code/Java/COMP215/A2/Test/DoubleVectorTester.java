import junit.framework.TestCase;
import java.util.Random;
import java.lang.Math;
import java.util.*;

/**
 * This abstract class serves as the basis from which all of the DoubleVector
 * implementations should be extended.  Most of the methods are abstract, but
 * this class does provide implementations of a couple of the DoubleVector methods.
 * See the DoubleVector interface for a description of each of the methods.
 */
abstract class ADoubleVector implements IDoubleVector {
  
  /** 
   * These methods are all abstract!
   */
  public abstract void addMyselfToHim (IDoubleVector addToThisOne) throws OutOfBoundsException;
  public abstract double getItem (int whichOne) throws OutOfBoundsException;
  public abstract void setItem (int whichOne, double setToMe) throws OutOfBoundsException;
  public abstract int getLength ();
  public abstract void addToAll (double addMe);
  public abstract double l1Norm ();
  public abstract void normalize ();
  
  /* These two methods re the only ones provided by the AbstractDoubleVector.
   * toString method constructs a string representation of a DoubleVector by adding each item to the string with < and > at the beginning and end of the string.
   */
  public String toString () {
    
    try {
      String returnVal = new String ("<");
      for (int i = 0; i < getLength (); i++) {
        Double curItem = getItem (i);
        // If the current index is not 0, add a comma and a space to the string
        if (i != 0)
          returnVal = returnVal + ", ";
        // Add the string representation of the current item to the string
        returnVal = returnVal + curItem.toString (); 
      }
      returnVal = returnVal + ">";
      return returnVal;
    
    } catch (OutOfBoundsException e) {
      
      System.out.println ("This is strange.  getLength() seems to be incorrect?");
      return new String ("<>");
    }
  }
  
  public long getRoundedItem (int whichOne) throws OutOfBoundsException {
    return Math.round (getItem (whichOne));      
  }
}


/**
 * This is the interface for a vector of double-precision floating point values.
 */
interface IDoubleVector {
  
  /** 
   * Adds the contents of this double vector to the other one.  
   * Will throw OutOfBoundsException if the two vectors
   * don't have exactly the same sizes.
   * 
   * @param addToHim       the double vector to be added to
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
   * This forces the absolute value of the sum of all of the items in the vector to be one, by 
   * dividing each item by the absolute value of the total over all items.
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
 * An interface to an indexed element with an integer index into its
 * enclosing container and data of parameterized type T.
 */
interface IIndexedData<T> {
  /**
   * Get index of this item.
   *
   * @return  index in enclosing container
   */
  int getIndex();

  /**
   * Get data.
   *
   * @return  data element
   */
  T   getData();
}

interface ISparseArray<T> extends Iterable<IIndexedData<T>> {
  /**
   * Add element to the array at position.
   * 
   * @param position  position in the array
   * @param element   data to place in the array
   */
  void put (int position, T element);

  /**
   * Get element at the given position.
   *
   * @param position  position in the array
   * @return          element at that position or null if there is none
   */
  T get (int position);

  /**
   * Create an iterator over the array.
   *
   * @return  an iterator over the sparse array
   */
  Iterator<IIndexedData<T>> iterator ();
}


/**
 * This is thrown when someone tries to access an element in a DoubleVector that is beyond the end of 
 * the vector.
 */
class OutOfBoundsException extends Exception {
  static final long serialVersionUID = 2304934980L;

  public OutOfBoundsException(String message) {
    super(message);
  }
}


/** 
 * Implementaiton of the DoubleVector interface that really just wraps up
 * an array and implements the DoubleVector operations on top of the array.
 */
class DenseDoubleVector extends ADoubleVector {
  
  // holds the data
  private double [] myData;
  
  // remembers what the the baseline is; that is, every value actually stored in
  // myData is a delta from this
  private double baselineData;
  
  /**
   * This creates a DenseDoubleVector having the specified length; all of the entries in the
   * double vector are set to have the specified initial value.
   * 
   * @param len             The length of the new double vector we are creating.
   * @param initialValue    The default value written into all entries in the vector
   */
  public DenseDoubleVector (int len, double initialValue) {
    
    // allocate the space
    myData = new double[len];
    
    // initialize the array
    for (int i = 0; i < len; i++) {
      myData[i] = 0.0;
    }
    
    // the baseline data is set to the initial value
    baselineData = initialValue;
  }
  
  public int getLength () {
    return myData.length;
  }
  
  /*
   * Method that calculates the L1 norm of the DoubleVector. 
   */
  public double l1Norm () {
    double returnVal = 0.0;
    for (int i = 0; i < myData.length; i++) {
      returnVal += Math.abs (myData[i] + baselineData);
    }
    return returnVal;
  }
  
  public void normalize ()  {
    double total = l1Norm ();
    for (int i = 0; i < myData.length; i++) {
      double trueVal = myData[i];
      trueVal += baselineData;
      myData[i] = trueVal / total - baselineData / total;
    }
    baselineData /= total;
  }
  
  public void addMyselfToHim (IDoubleVector addToThisOne) throws OutOfBoundsException {

    if (getLength() != addToThisOne.getLength()) {
      throw new OutOfBoundsException("vectors are different sizes");
    }
    
    // easy... just do the element-by-element addition
    for (int i = 0; i < myData.length; i++) {
      double value = addToThisOne.getItem (i);
      value += myData[i];
      addToThisOne.setItem (i, value);
    }
    addToThisOne.addToAll (baselineData);
  }
  
  public double getItem (int whichOne) throws OutOfBoundsException {
    
    if (whichOne >= myData.length) {      
      throw new OutOfBoundsException ("index too large in getItem");
    }
    
    return myData[whichOne] + baselineData;
  }
  
  public void setItem (int whichOne, double setToMe) throws OutOfBoundsException {
    
    if (whichOne >= myData.length) {
      throw new OutOfBoundsException ("index too large in setItem");
    }    

    myData[whichOne] = setToMe - baselineData;
  }
  
  public void addToAll (double addMe) {
    baselineData += addMe;
  }
  
}


/**
 * Implementation of the DoubleVector that uses a sparse representation (that is,
 * not all of the entries in the vector are explicitly represented).  All non-default
 * entires in the vector are stored in a HashMap.
 */
class SparseDoubleVector extends ADoubleVector {
  
  // this stores all of the non-default entries in the sparse vector
  private ISparseArray<Double> nonEmptyEntries;
  
  // everyone in the vector has this value as a baseline; the actual entries 
  // that are stored are deltas from this
  private double baselineValue;
  
  // how many entries there are in the vector
  private int myLength;
 
  /**
   * Creates a new DoubleVector of the specified length with the spefified initial value.
   * Note that since this uses a sparse represenation, putting the inital value in every
   * entry is efficient and uses no memory.
   * 
   * @param len            the length of the vector we are creating
   * @param initalValue    the initial value to "write" in all of the vector entries
   */
  public SparseDoubleVector (int len, double initialValue) {
    myLength = len;
    baselineValue = initialValue;
    nonEmptyEntries = new SparseArray<Double>();
  }
  
  public void normalize () {
  
    double total = l1Norm ();
    for (IIndexedData<Double> el : nonEmptyEntries) {
      Double value = el.getData();
      int position = el.getIndex();
      double newValue = (value + baselineValue) / total;
      value = newValue - (baselineValue / total);
      nonEmptyEntries.put(position, value);
    }
    
    baselineValue /= total;
  }
  
  public double l1Norm () {
    
    // iterate through all of the values
    double returnVal = 0.0;
    int nonEmptyEls = 0;
    for (IIndexedData<Double> el : nonEmptyEntries) {
      returnVal += Math.abs (el.getData() + baselineValue);
      nonEmptyEls += 1;
    }
    
    // and add in the total for everyone who is not explicitly represented
    returnVal += Math.abs ((myLength - nonEmptyEls) * baselineValue);
    return returnVal;
  }
  
  public void addMyselfToHim (IDoubleVector addToHim) throws OutOfBoundsException {
    // make sure that the two vectors have the same length
    if (getLength() != addToHim.getLength ()) {
      throw new OutOfBoundsException ("unequal lengths in addMyselfToHim");
    }
  
    // add every non-default value to the other guy
    for (IIndexedData<Double> el : nonEmptyEntries) {
      double myVal = el.getData();
      int curIndex = el.getIndex();
      myVal += addToHim.getItem (curIndex);
      addToHim.setItem (curIndex, myVal);
    }
    
    // and add my baseline in
    addToHim.addToAll (baselineValue);
  }
  
  public void addToAll (double addMe) {
    baselineValue += addMe;
  }
  
  public double getItem (int whichOne) throws OutOfBoundsException {
        
    // make sure we are not out of bounds
    if (whichOne >= myLength || whichOne < 0) {
      throw new OutOfBoundsException ("index too large in getItem");
    }
    
    // now, look the thing up
    Double myVal = nonEmptyEntries.get (whichOne);
    if (myVal == null) {
      return baselineValue;
    } else {
      return myVal + baselineValue;
    }
  }
  
  public void setItem (int whichOne, double setToMe) throws OutOfBoundsException {

    // make sure we are not out of bounds
    if (whichOne >= myLength || whichOne < 0) {
      throw new OutOfBoundsException ("index too large in setItem");
    }
    
    // try to put the value in
    nonEmptyEntries.put (whichOne, setToMe - baselineValue);
  }
  
  public int getLength () {
    return myLength;
  }
}



/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
public class DoubleVectorTester extends TestCase {
  
  /**
   * In this part of the code we have a number of helper functions
   * that will be used by the actual test functions to test specific
   * functionality.
   */
  
  /**
   * This is used all over the place to check if an observed value is
   * close enough to an expected double value
   */
  private void checkCloseEnough (double observed, double goal, int pos) {
    if (!(observed >= goal - 1e-8 - Math.abs (goal * 1e-8) && 
          observed <= goal + 1e-8 + Math.abs (goal * 1e-8)))
      if (pos != -1) 
        fail ("Got " + observed + " at pos " + pos + ", expected " + goal);
      else
        fail ("Got " + observed + ", expected " + goal);
  }
  
  /** 
   * This adds a bunch of data into testMe, then checks (and returns)
   * the l1norm
   */
  private double l1NormTester (IDoubleVector testMe, double init) {
  
    // This will by used to scatter data randomly
    
    // Note we don't want test cases to share the random number generator, since this
    // will create dependencies among them
    Random rng = new Random (12345);
    
    // pick a bunch of random locations and repeatedly add an integer in
    double total = 0.0;
    int pos = 0;
    while (pos < testMe.getLength ()) {
      
      // there's a 20% chance we keep going
      if (rng.nextDouble () > .2) {
        pos++;
        continue;
      }
      
      // otherwise, add a value in
      try {
        double current = testMe.getItem (pos);
        testMe.setItem (pos, current + pos);
        total += pos;
      } catch (OutOfBoundsException e) {
        fail ("Got an out of bounds exception on set/get pos " + pos + " not expecting one!\n");
      }
    }
    
    // now test the l1 norm
    double expected = testMe.getLength () * init + total;
    checkCloseEnough (testMe.l1Norm (), expected, -1);
    return expected;
  }
  
  /**
   * This does a large number of setItems to an array of double vectors,
   * and then makes sure that all of the set values are still there
   */
  private void doLotsOfSets (IDoubleVector [] allMyVectors, double init) {
    
    int numVecs = allMyVectors.length;
    
    // put data in exectly one array in each dimension
    for (int i = 0; i < allMyVectors[0].getLength (); i++) {
      
      int whichOne = i % numVecs;
      try {
        allMyVectors[whichOne].setItem (i, 1.345);
      } catch (OutOfBoundsException e) {
        fail ("Got an out of bounds exception on set/get pos " + whichOne + " not expecting one!\n");
      }
    }
    
    // now check all of that data
    // put data in exectly one array in each dimension
    for (int i = 0; i < allMyVectors[0].getLength (); i++) {
      
      int whichOne = i % numVecs;
      for (int j = 0; j < numVecs; j++) {
        try {
          if (whichOne == j) {
            checkCloseEnough (allMyVectors[j].getItem (i), 1.345, j);
          } else {
            checkCloseEnough (allMyVectors[j].getItem (i), init, j); 
          }        
        } catch (OutOfBoundsException e) {
          fail ("Got an out of bounds exception on set/get pos " + whichOne + " not expecting one!\n");
        }
      }
    }
  }
  
  /**
   * This sets only the first value in a double vector, and then checks
   * to see if only the first vlaue has an updated value.
   */
  private void trivialGetAndSet (IDoubleVector testMe, double init) {
    try {
      
      // set the first item
      testMe.setItem (0, 11.345);
      
      // now retrieve and test everything except for the first one
      for (int i = 1; i < testMe.getLength (); i++) {
        checkCloseEnough (testMe.getItem (i), init, i);
      }
      
      // and check the first one
      checkCloseEnough (testMe.getItem (0), 11.345, 0);
     
    } catch (OutOfBoundsException e) {
      fail ("Got an out of bounds exception on set/get... not expecting one!\n");
    }
  }
  
  /**
   * This uses the l1NormTester to set a bunch of stuff in the input
   * DoubleVector.  It then normalizes it, and checks to see that things
   * have been normalized correctly.
   */
  private void normalizeTester (IDoubleVector myVec, double init) {

    // get the L1 norm, and make sure it's correct
    double result = l1NormTester (myVec, init);
    
    try {
      // now get an array of ratios
      double [] ratios = new double[myVec.getLength ()];
      for (int i = 0; i < myVec.getLength (); i++) {
        ratios[i] = myVec.getItem (i) / result;
      }
    
      // and do the normalization on myVec
      myVec.normalize ();
    
      // now check it if it is close enough to an expected double value
      for (int i = 0; i < myVec.getLength (); i++) {
        checkCloseEnough (ratios[i], myVec.getItem (i), i);
      } 
      
      // and make sure the length is one
      checkCloseEnough (myVec.l1Norm (), 1.0, -1);
    
    } catch (OutOfBoundsException e) {
        fail ("Got an out of bounds exception on set/get... not expecting one!\n");
    } 
  }
  
  /**
   * Here we have the various test functions, organized in increasing order of difficulty.
   * The code for most of them is quite short, and self-explanatory.
   */
 
  public void testSingleDenseSetSize1 () {
    int vecLen = 1;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 45.67);
    assertEquals (myVec.getLength (), vecLen);
    trivialGetAndSet (myVec, 45.67);
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testSingleSparseSetSize1 () {
    int vecLen = 1;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 345.67);
    assertEquals (myVec.getLength (), vecLen);
    trivialGetAndSet (myVec, 345.67);
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testSingleDenseSetSize1000 () {
    int vecLen = 1000;
    IDoubleVector myVec = new DenseDoubleVector (vecLen, 245.67);
    assertEquals (myVec.getLength (), vecLen);
    trivialGetAndSet (myVec, 245.67);
    assertEquals (myVec.getLength (), vecLen);
  }
  
  //initialValue: 145.67
  public void testSingleSparseSetSize1000 () {
    int vecLen = 1000;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 145.67);
    assertEquals (myVec.getLength (), vecLen);
    trivialGetAndSet (myVec, 145.67);
    assertEquals (myVec.getLength (), vecLen);
  }  
  
  public void testLotsOfDenseSets () {
  
    int numVecs = 100;
    int vecLen = 10000;
    IDoubleVector [] allMyVectors = new DenseDoubleVector [numVecs];
    for (int i = 0; i < numVecs; i++) {
      allMyVectors[i] = new DenseDoubleVector (vecLen, 2.345);
    }
    
    assertEquals (allMyVectors[0].getLength (), vecLen);
    doLotsOfSets (allMyVectors, 2.345);
    assertEquals (allMyVectors[0].getLength (), vecLen);
  }
  
  public void testLotsOfSparseSets () {
  
    int numVecs = 100;
    int vecLen = 10000;
    IDoubleVector [] allMyVectors = new SparseDoubleVector [numVecs];
    for (int i = 0; i < numVecs; i++) {
      allMyVectors[i] = new SparseDoubleVector (vecLen, 2.345);
    }
 
    assertEquals (allMyVectors[0].getLength (), vecLen);
    doLotsOfSets (allMyVectors, 2.345);
    assertEquals (allMyVectors[0].getLength (), vecLen);
  }
  
  public void testLotsOfDenseSetsWithAddToAll () {
  
    int numVecs = 100;
    int vecLen = 10000;
    IDoubleVector [] allMyVectors = new DenseDoubleVector [numVecs];
    for (int i = 0; i < numVecs; i++) {
      allMyVectors[i] = new DenseDoubleVector (vecLen, 0.0);
      allMyVectors[i].addToAll (2.345);
    }
    
    assertEquals (allMyVectors[0].getLength (), vecLen);
    doLotsOfSets (allMyVectors, 2.345);
    assertEquals (allMyVectors[0].getLength (), vecLen);
  }
  
  public void testLotsOfSparseSetsWithAddToAll () {
  
    int numVecs = 100;
    int vecLen = 10000;
    IDoubleVector [] allMyVectors = new SparseDoubleVector [numVecs];
    for (int i = 0; i < numVecs; i++) {
      allMyVectors[i] = new SparseDoubleVector (vecLen, 0.0);
      allMyVectors[i].addToAll (-12.345);
    }
 
    assertEquals (allMyVectors[0].getLength (), vecLen);
    doLotsOfSets (allMyVectors, -12.345);
    assertEquals (allMyVectors[0].getLength (), vecLen);
  }
  
  public void testL1NormDenseShort () {
    int vecLen = 10;
    IDoubleVector myVec = new DenseDoubleVector (vecLen, 45.67);
    assertEquals (myVec.getLength (), vecLen);
    l1NormTester (myVec, 45.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testL1NormDenseLong () {
    int vecLen = 100000;
    IDoubleVector myVec = new DenseDoubleVector (vecLen, 45.67);
    assertEquals (myVec.getLength (), vecLen);
    l1NormTester (myVec, 45.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testL1NormSparseShort () {
    int vecLen = 10;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 45.67);
    assertEquals (myVec.getLength (), vecLen);
    l1NormTester (myVec, 45.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testL1NormSparseLong () {
    int vecLen = 100000;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 45.67);
    assertEquals (myVec.getLength (), vecLen);
    l1NormTester (myVec, 45.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testRounding () {
    
    // Note we don't want test cases to share the random number generator, since this
    // will create dependencies among them
    Random rng = new Random (12345);
    
    // create the vectors
    int vecLen = 100000;
    IDoubleVector myVec1 = new DenseDoubleVector (vecLen, 55.0);
    IDoubleVector myVec2 = new SparseDoubleVector (vecLen, 55.0);
    
    // put values with random additional precision in
    for (int i = 0; i < vecLen; i++) {
      double valToPut = i + (0.5 - rng.nextDouble ()) * .9;
      try {
        myVec1.setItem (i, valToPut);
        myVec2.setItem (i, valToPut);
      } catch (OutOfBoundsException e) {
        fail ("not expecting an out-of-bounds here");
      }
    }
    
    // and extract those values
    for (int i = 0; i < vecLen; i++) {
      try {
        checkCloseEnough (myVec1.getRoundedItem (i), i, i);
        checkCloseEnough (myVec2.getRoundedItem (i), i, i);
      } catch (OutOfBoundsException e) {
        fail ("not expecting an out-of-bounds here");
      }
    }
  }  
  
  public void testOutOfBounds () {
   
    int vecLen = 1000;
    SparseDoubleVector vec1 = new SparseDoubleVector (vecLen, 0.0);
    SparseDoubleVector vec2 = new SparseDoubleVector (vecLen + 1, 0.0);
    
    try {
      vec1.getItem (-1);
      fail ("Missed bad getItem #1");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec1.getItem (vecLen);
      fail ("Missed bad getItem #2");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec1.setItem (-1, 0.0);
      fail ("Missed bad setItem #3");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec1.setItem (vecLen, 0.0);
      fail ("Missed bad setItem #4");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec2.getItem (-1);
      fail ("Missed bad getItem #5");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec2.getItem (vecLen + 1);
      fail ("Missed bad getItem #6");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec2.setItem (-1, 0.0);
      fail ("Missed bad setItem #7");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec2.setItem (vecLen + 1, 0.0);
      fail ("Missed bad setItem #8");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec1.addMyselfToHim (vec2);
      fail ("Missed bad add #9");
    } catch (OutOfBoundsException e) {}
    
    try {
      vec2.addMyselfToHim (vec1);
      fail ("Missed bad add #10");
    } catch (OutOfBoundsException e) {}
  }
  
  /**
   * This test creates 100 sparse double vectors, and puts a bunch of data into them
   * pseudo-randomly.  Then it adds each of them, in turn, into a single dense double
   * vector, which is then tested to see if everything adds up.
   */
  public void testLotsOfAdds() {
 
    // This will by used to scatter data randomly into various sparse arrays
    // Note we don't want test cases to share the random number generator, since this
    // will create dependencies among them
    Random rng = new Random (12345);
 
    IDoubleVector [] allMyVectors = new IDoubleVector [100];
    for (int i = 0; i < 100; i++) {
      allMyVectors[i] = new SparseDoubleVector (10000, i * 2.345);
    }
    
    // put data in each dimension
    for (int i = 0; i < 10000; i++) {
      
      // randomly choose 20 dims to put data into
      for (int j = 0; j < 20; j++) {
        int whichOne = (int) Math.floor (100.0 * rng.nextDouble ());
        try {
          allMyVectors[whichOne].setItem (i, allMyVectors[whichOne].getItem (i) + i * 2.345);
        } catch (OutOfBoundsException e) {
          fail ("Got an out of bounds exception on set/get pos " + whichOne + " not expecting one!\n");
        }
      }
    }
    
    // now, add the data up
    DenseDoubleVector result = new DenseDoubleVector (10000, 0.0);
    for (int i = 0; i < 100; i++) {
      try {
        allMyVectors[i].addMyselfToHim (result);
      } catch (OutOfBoundsException e) {
        fail ("Got an out of bounds exception adding two vectors, not expecting one!");
      }
    }
    
    // and check the final result
    for (int i = 0; i < 10000; i++) {
      double expectedValue = (i * 20.0 + 100.0 * 99.0 / 2.0) * 2.345;
      try {
        checkCloseEnough (result.getItem (i), expectedValue, i);
      } catch (OutOfBoundsException e) {
        fail ("Got an out of bounds exception on getItem, not expecting one!");
      }
    }
  }
  
  public void testNormalizeDense () {
    int vecLen = 100000;
    IDoubleVector myVec = new DenseDoubleVector (vecLen, 55.67);
    assertEquals (myVec.getLength (), vecLen);
    normalizeTester (myVec, 55.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
  
  public void testNormalizeSparse () {
    int vecLen = 100000;
    IDoubleVector myVec = new SparseDoubleVector (vecLen, 55.67);
    assertEquals (myVec.getLength (), vecLen);
    normalizeTester (myVec, 55.67); 
    assertEquals (myVec.getLength (), vecLen);
  }
}
