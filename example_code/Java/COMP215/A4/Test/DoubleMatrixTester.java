import junit.framework.TestCase;
import java.util.Random;
import java.util.*;

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

/**
 * This interface corresponds to a sparse implementation of a matrix
 * of double-precision values.
 */
interface IDoubleMatrix {
 
  /** 
   * This returns the i^th row in the matrix.  Note that the row that
   * is returned may contain one or more references to data that are
   * actually contained in the matrix, so if the caller modifies this
   * row, it could end up modifying the row in the underlying matrix in
   * an unpredicatble way.  If i exceeds the number of rows in the matrix
   * or it is less than zero, an OutOfBoundsException is thrown.
   */
  IDoubleVector getRow (int i) throws OutOfBoundsException;
  
  /** 
   * This returns the j^th column in the matrix.  All of the comments
   * above regarding getRow apply.  If j exceeds the number of columns in the
   * matrix or it is less than zero, an OutOfBoundsException is thrown.
   */
  IDoubleVector getColumn (int j) throws OutOfBoundsException;
  
  /**
   * This sets the i^th row of the matrix.  After the row is inserted into
   * the matrix, the matrix "owns" the row and it is free to do whatever it
   * wants to it, including modifying the row.  If i exceeds the number of rows
   * in the matrix or it is less than zero, an OutOfBoundsException is thrown.
   */
  void setRow (int i, IDoubleVector setToMe) throws OutOfBoundsException;
  
  /**
   * This sets the j^th column of the matrix.  All of the comments above for
   * the "setRow" method apply to "setColumn".  If j exceeds the number of columns
   * in the matrix or it is less than zero, an OutOfBoundsException is thrown.
   */
  void setColumn (int j, IDoubleVector setToMe) throws OutOfBoundsException;
  
  /**
   * Returns the entry in the i^th row and j^th column in the matrix.
   * If i or j are less than zero, or if j exceeds the number of columns
   * or i exceeds the number of rows, then an OutOfBoundsException is thrown.
   */
  double getEntry (int i, int j) throws OutOfBoundsException;
  
  /**
   * Sets the entry in the i^th row and j^th column in the matrix.
   * If i or j are less than zero, or if j exceeds the number of columns
   * or i exceeds the number of rows, then an OutOfBoundsException is thrown.
   */
  void setEntry (int i, int j, double setToMe) throws OutOfBoundsException;
  
  /**
   * Adds this particular IDoubleMatrix to the parameter.  Returns an
   * OutOfBoundsException if the two don't match up in terms of their dimensions.
   */
  void addMyselfToHim (IDoubleMatrix toMe) throws OutOfBoundsException;

  /** 
   * Sums all of the rows of this IDoubleMatrix.  It is the equivalent of saying:
   *
   * SparseDoubleVector accum = new SparseDoubleVector (myMatrix.getNumColumns (), 0.0);
   * for (int i = 0; i < myMatrix.getNumRows (); i++) {
   *   myMatrix.getRow (i).addMyselfToHim (accum);
   * }
   * return accum;
   *
   * Note, however, that the implementation should be more efficient than this if you
   * have a row major matrix... then, it should run on O(m) time, where m is the number
   * of non-empty rows.
   */
  IDoubleVector sumRows ();

  /**
   * Sums all of the columns of this IDoubleMatrix.  Returns the result.
   */
  IDoubleVector sumColumns ();

  /**
   * Returns the number of rows in the matrix.
   */
  int getNumRows ();
  
  /**
   * Returns the number of columns in the matrix.
   */
  int getNumColumns ();
  
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

abstract class ADoubleMatrix implements IDoubleMatrix {
 
  /**
   * The basic idea here is to put everything in the anstract class.  Then the two
   * concrete classes (row major and column major) just call these functions... the
   * only difference between row major and column major is that they call different
   * operations
   */
  private ISparseArray <IDoubleVector> myData;
  private int numCols;
  private int numRows;
  private double backValue;
  
  protected ADoubleMatrix (int numRowsIn, int numColsIn, double backValueIn) {
    numCols = numColsIn;
    numRows = numRowsIn;
    backValue = backValueIn;
    myData = new LinearSparseArray <IDoubleVector> (10);
    if (numCols < 0 || numRows < 0)
      throw new RuntimeException ("Bad input index");
  }
  
  protected IDoubleVector getRowABS (int j) throws OutOfBoundsException {
    if (j >= numRows || j < 0)
      throw new OutOfBoundsException ("row out of bounds in getRowABS");
    
    IDoubleVector returnVal = myData.get (j);
    if (returnVal == null) {
      returnVal = new SparseDoubleVector (numCols, backValue);  
    }
    return returnVal;
  }
  
  protected IDoubleVector getColumnABS (int i) throws OutOfBoundsException {
    if (i >= numCols || i < 0)
      throw new OutOfBoundsException ("col out of bounds in getColABS");
    
    IDoubleVector returnVal = new SparseDoubleVector (numRows, backValue);
    
    for (int row = 0; row < numRows; row++) {
      IDoubleVector myRow = myData.get (row);
      if (myRow != null) {
        returnVal.setItem (row, myRow.getItem (i));
      }
    }
    return returnVal;
  }
  
  protected void setRowABS (int j, IDoubleVector setToMe) throws OutOfBoundsException {
    if (j >= numRows || j < 0 || setToMe.getLength () != numCols)
      throw new OutOfBoundsException ("row out of bounds in setRowABS");
    myData.put (j, setToMe);
  }
  
  protected void setColumnABS (int i, IDoubleVector setToMe) throws OutOfBoundsException {
    if (i >= numCols || i < 0 || setToMe.getLength () != numRows)
      throw new OutOfBoundsException ("col out of bounds in setColumnABS");
    
    for (int row = 0; row < numRows; row++) {
      IDoubleVector myRow = myData.get (row);
      if (myRow == null) {
        myRow = new SparseDoubleVector (numCols, backValue);
        myData.put (row, myRow);
      }
      myRow.setItem (i, setToMe.getItem (row));
    } 
  }
  
  protected double getEntryABS (int j, int i) throws OutOfBoundsException {
    if (j >= numRows || j < 0)
      throw new OutOfBoundsException ("row out of bounds in getEntryABS");
    if (i >= numCols || i < 0)
      throw new OutOfBoundsException ("col out of bounds in getEntryABS");
    IDoubleVector myRow = myData.get (j);
    if (myRow == null) {
      return backValue;  
    }
    return myRow.getItem (i);
  }
  
  public IDoubleVector sumRows () {
    IDoubleVector sum = new DenseDoubleVector (getNumColumns (), 0.0);
    for (int i = 0; i < getNumRows (); i++) {
      try {
        IDoubleVector curRow = getRow (i);
        curRow.addMyselfToHim (sum);
      } catch (OutOfBoundsException e) {throw new RuntimeException (e);}
    }
    return sum;
  }
  
  public IDoubleVector sumColumns () {
    IDoubleVector sum = new DenseDoubleVector (getNumRows (), 0.0);
    for (int i = 0; i < getNumColumns (); i++) {
      try {
        IDoubleVector curCol = getColumn (i);
        curCol.addMyselfToHim (sum);
      } catch (OutOfBoundsException e) {throw new RuntimeException (e);}
    }
    return sum;
  }
  
  protected void setEntryABS (int j, int i, double setToMe) throws OutOfBoundsException {
    if (j >= numRows || j < 0)
      throw new OutOfBoundsException ("row out of bounds in setEntryABS");
    if (i >= numCols || i < 0)
      throw new OutOfBoundsException ("col out of bounds in setEntryABS");
    IDoubleVector myRow = myData.get (j);
    if (myRow == null) {
      myRow = new SparseDoubleVector (numCols, backValue);
      myData.put (j, myRow);
    }
    myRow.setItem (i, setToMe);
  }
  
  protected int getNumRowsABS () {
    return numRows;
  }
  
  protected int getNumColumnsABS () {
    return numCols;
  }
  
  /**
   * Theare are the various operations that will be implemented in the concrete classes
   */
  abstract public IDoubleVector getRow (int j) throws OutOfBoundsException;
  abstract public IDoubleVector getColumn (int i) throws OutOfBoundsException;
  abstract public void setRow (int j, IDoubleVector setToMe) throws OutOfBoundsException;
  abstract public void setColumn (int i, IDoubleVector setToMe) throws OutOfBoundsException;
  abstract public double getEntry (int i, int j) throws OutOfBoundsException;
  abstract public void setEntry (int i, int j, double setToMe) throws OutOfBoundsException;
  abstract public int getNumRows ();
  abstract public int getNumColumns ();
}


class ColumnMajorDoubleMatrix extends ADoubleMatrix {
  
  public ColumnMajorDoubleMatrix (int numRows, int numCols, double defaultVal) {
    super (numCols, numRows, defaultVal);  
  }
  
  public IDoubleVector getRow (int j) throws OutOfBoundsException {
    return getColumnABS (j);  
  }
  
  public IDoubleVector getColumn (int i) throws OutOfBoundsException {
    return getRowABS (i);  
  }
  
  public void setRow (int j, IDoubleVector setToMe) throws OutOfBoundsException {
    setColumnABS (j, setToMe);  
  }
  
  public void addMyselfToHim (IDoubleMatrix toMe) throws OutOfBoundsException {
    int numCols = getNumColumns ();
    for (int i = 0; i < numCols; i++) {
      IDoubleVector curCol = getColumn (i);
      IDoubleVector hisCol = toMe.getColumn (i);
      curCol.addMyselfToHim (hisCol);
      toMe.setColumn (i, hisCol);
    }
  }
    
  public void setColumn (int i, IDoubleVector setToMe) throws OutOfBoundsException {
    setRowABS (i, setToMe);  
  }
  
  public double getEntry (int i, int j) throws OutOfBoundsException {
    return getEntryABS (j, i);  
  }
  
  public void setEntry (int i, int j, double setToMe) throws OutOfBoundsException {
    setEntryABS (j, i, setToMe);
  }
  
  public int getNumRows () {
    return getNumColumnsABS ();  
  }
  
  public int getNumColumns () {
    return getNumRowsABS ();
  }
}


class RowMajorDoubleMatrix extends ADoubleMatrix {
  
  public RowMajorDoubleMatrix (int numRows, int numCols, double defaultVal) {
    super (numRows, numCols, defaultVal);  
  }
    
  public IDoubleVector getRow (int j) throws OutOfBoundsException {
    return getRowABS (j);  
  }
  
  public IDoubleVector getColumn (int i) throws OutOfBoundsException {
    return getColumnABS (i);  
  }
  
  public void setRow (int j, IDoubleVector setToMe) throws OutOfBoundsException {
    setRowABS (j, setToMe);  
  }
  
  public void addMyselfToHim (IDoubleMatrix toMe) throws OutOfBoundsException {
    int numRows = getNumRows ();
    for (int i = 0; i < numRows; i++) {
      IDoubleVector curRow = getRow (i);
      IDoubleVector hisRow = toMe.getRow (i);
      curRow.addMyselfToHim (hisRow);
      toMe.setRow (i, hisRow);
    }
  }
    
  public void setColumn (int i, IDoubleVector setToMe) throws OutOfBoundsException {
    setColumnABS (i, setToMe);  
  }
  
  public double getEntry (int i, int j) throws OutOfBoundsException {
    return getEntryABS (i, j);  
  }
  
  public void setEntry (int i, int j, double setToMe) throws OutOfBoundsException {
    setEntryABS (i, j, setToMe);
  }
  
  public int getNumRows () {
    return getNumRowsABS ();  
  }
  
  public int getNumColumns () {
    return getNumColumnsABS ();
  }
}
/**
 * A JUnit test case class for testing an implementation of an IDoubleMatrix. Every method starting with the word "test" will be called when running the test with JUnit.
 */
public class DoubleMatrixTester extends TestCase {
// Constants to contain the boundary values we will test for.
  private final double MAX_DOUBLE = Double.MAX_VALUE;
  private final double MIN_DOUBLE = Double.MIN_VALUE;
  private final double POSITIVE_INFINITY = Double.POSITIVE_INFINITY;
  private final double NEGATIVE_INFINITY = Double.NEGATIVE_INFINITY;
  
  // An array of doubles of test values that we test every method on.  These include the boundary values from above, 0.0, average values, decimals, negative, maximum, minimum, positive and negative inifinity.
  private final double[] testValues = {0.0, 1.0, -1.0, -100.0, 100.0, 15.1610912461267127317, -2372.3616123612361237132, MAX_DOUBLE, MIN_DOUBLE, POSITIVE_INFINITY, NEGATIVE_INFINITY};
  // An array of doubles of initial values that we test every method on.  These include the boundary values from above, 0.0, average values, and decimals.
  // Note that Positive and Negative infinity is excluded here.  This is due to a problem with the initialization of SparseDoubleVectors.  If we initialize a 
  // sparse double vector with infinity, changing it from infinity will result in NaN.  Note that NaN compared to NaN always results in false.
  private final double[] initValues = {0.0, 1.0, -1.0, -100.0, 100.0, 15.1610912461267127317, -2372.3616123612361237132, MAX_DOUBLE, MIN_DOUBLE};  
  
  // Here, we have a different set of values for the "set values".  We have to use these values as opposed to the init values above due to precision and boundary issues.
  // For the boundary issues, when we utilize MAX_DOUBLE, but then set a value to a negative number, the SpareDoubleVector will store a number that is less than the minimum possible number
  // at this point.  This will result in an overflow, and loop back around to 0.  A similar thing happens with MIN_DOUBLE and positive values.
  // On the other hand, when we test with numbers with many decimals, this leads to losses in precision when we set values.  Therefore, these small losses in precision will evaluate
  // to a miss when we compare it.  
  private final double[] setValues = {0.0, -1.0, -6000000.0, 5000000000000.0};  
  // An array of integers that represent indices we utilize when testing for failures (when we are catching OutOfBoundaryExceptions)
  private final int[] failIndices = {-100, -1, 1000};
  // An array of integers that represent indices that work when we test for failures (when we are catching OutOfBoundaryExceptions)
  private final int[] validIndices = {0, 99};
  // An array of integers that represent lengths of columns [rows] that would not be accepted by an array of height [length] 100.
  private final int[] failLengths = {1, 99, 1000};
 /**
  * This is used all over the place to check if an observed value is close
  * enough to an expected double value
  */
 private void checkCloseEnough(double observed, double goal) {
  if (!(observed > goal - 0.000001 && 
        observed < goal + 0.000001)) {
   fail("Got " + observed + ", expected " + goal);
  }
 }
 
 /**
  * Tests that the matrix's size (number of columns, number of rows) is set properly.
  */
 public void testMatrixSize() {
  
  for (int i = 0; i < 100; i += 10) {
   IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(i*2, i, 5.0);
   assertEquals("Row major matrix size (col) failed to match", i, rMatrix.getNumColumns());
   assertEquals("Row major matrix size (rows) failed to match", i*2, rMatrix.getNumRows());

   IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(i*2, i, 5.0);
   assertEquals("Column major matrix size (col) failed to match", i, cMatrix.getNumColumns());
   assertEquals("Column major matrix size (rows) failed to match", i*2, cMatrix.getNumRows());
  }
  
  throwsCorrectExceptionWithInitialization(-1, 5);
  throwsCorrectExceptionWithInitialization(5, -1);
  throwsCorrectExceptionWithInitialization(-3, -3);
 }
 
 /**
  * Tests if the matrix's constructor does its job properly with guaranteeing valid inputs for 
  * the number of columns and number of rows.
  * Precondition: numCols or numRows should be invalid (less than 0)
  * 
  * @param numCols number of columns in the matrix to initialize
  * @param numRows number of rows in the matrix to initialize
  */
 private void throwsCorrectExceptionWithInitialization(int numCols, int numRows) {
  boolean errorNotThrown = true;
  
  try {
   // try to initialize matrices of invalid proportions, which should throw an exception
   IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
   IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
  } catch (Exception e) {
   errorNotThrown = false;
  }

  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds or IllegalArgument exception to be thrown.");
  }
 }
 
 /**
  * Test that every element of a newly initialized matrix has the same initial value.
  */
 public void testMatrixInitialValue() {
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(4, 3, 5.0);
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(4, 3, 5.0);
  
  checkAllMatrixElementsEqualTo(rMatrix, 3, 4, 5.0);
  checkAllMatrixElementsEqualTo(cMatrix, 3, 4, 5.0);
 }
 
 /**
  * Checks that all elements of a matrix are equal to the same value.
  * @param matrix -  matrix to be checked
  * @param numCols - number of columns of the matrix
  * @param numRows - number of rows of the matrix
  * @param sameValue - value that all elements of the matrix should be equal to
  */
 private void checkAllMatrixElementsEqualTo(IDoubleMatrix matrix, int numCols, int numRows, double sameValue) {
  for (int i = 0; i < numCols; i++) {
   for (int j = 0; j < numRows; j++) {
    try {
     if (matrix.getEntry(j, i) != sameValue) {
      fail("The initial value of the matrix was not set correctly.");
     }
    } catch (OutOfBoundsException e) {
     fail("Encountered an out of bounds exception when doing a valid get operation.");
    }
   }
  }
 }
 
 /**
  * Helper method that tests if a primitive array and an IDoubleVector have the same elements.
  * Makes use of checkCloseEnough to do the comparison between doubles.
  * @param expected array of expected values
  * @param actual IDoubleVector of actual values
  */
 private void checkListsAreEqual(double[] expected, IDoubleVector actual) {
  for (int i = 0; i < expected.length; i++) {
   try {
    checkCloseEnough(actual.getItem(i), expected[i]);
   } catch (OutOfBoundsException e) {
    fail("Encountered an out of bounds exception when doing a valid get operation.");
   }
  }
 }
 
 /**
  * Tests a simple setting and getting n a matrix.
  */
 public void testSimpleSetAndGet() {
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(10, 10, 1.0);
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(10, 10, 1.0);

  trySimpleSetAndGetOnMatrix(rMatrix);
  trySimpleSetAndGetOnMatrix(cMatrix);
 }

 /**
  * Used to test a simple setting and getting of an element in a matrix.
  * Checks for out-of-bounds access as well.
  * @param matrix matrix to be tested
  */
 public void trySimpleSetAndGetOnMatrix(IDoubleMatrix matrix) {
  try {
   // is a valid entry index (colNum, rowNum) - this operation shuold succeed
   matrix.setEntry(5, 6, 9999);
  } catch (OutOfBoundsException e) {
   fail("Encountered an out of bounds exception when doing a valid set operation.");
  }
  
  // check out of bounds accesses on both dimensions with setEntry()
  throwsCorrectExceptionWithSet(matrix, -10, 5, 100);
  throwsCorrectExceptionWithSet(matrix, 5, -10, 100);
  throwsCorrectExceptionWithSet(matrix, 17, 6, 100); 
  throwsCorrectExceptionWithSet(matrix, 6, 17, 100);
  
  try {
   checkCloseEnough(matrix.getEntry(5, 6), 9999);
  } catch (OutOfBoundsException e) {
   fail("Encountered an out of bounds exception when doing a valid get operation.");

  }
  
  // check out of bounds accesses on both dimensions with getEntry()
  throwsCorrectExceptionWithGet(matrix, -10, 5);
  throwsCorrectExceptionWithGet(matrix, 5, -10);
  throwsCorrectExceptionWithGet(matrix, 17, 6); 
  throwsCorrectExceptionWithGet(matrix, 6, 17);
 }
 
 /**
  * Checks that an exception is thrown when an invalid index is accessed in the given matrix
  * with setEntry().
  * Precondition: should be called with indices that are out of bounds for the given matrix,
  * as defined in the specification.
  * @param matrix
  * @param i column in matrix to be accessed
  * @param j row in matrix to be accessed
  * @param value value to set the specified test entry as
  */
 private void throwsCorrectExceptionWithSet(IDoubleMatrix matrix, int i, int j, double value) {
  boolean errorNotThrown = true;
  // try to set entry at an invalid set of indexes; setEntry should throw an error
  try {
   matrix.setEntry(j, i, value);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  }

  // if no error was detected, then fail the test
  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds exception to be thrown.");
  }
 }
 
 /**
  * Checks that an exception is thrown when an invalid index is accessed in the given matrix
  * with getEntry().
  * Precondition: should be called with indices that are out of bounds for the given matrix,
  * as defined in the specification.
  * @param matrix
  * @param i column in matrix to be accessed
  * @param j row in matrix to be accessed
  */
 private void throwsCorrectExceptionWithGet(IDoubleMatrix matrix, int i, int j) {
  boolean errorNotThrown = true;
  // try to set entry at an invalid set of indexes; setEntry should throw an error
  try {
   matrix.getEntry(j, i);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  }

  // if no error was detected, then fail the test
  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds exception to be thrown.");
  }
 }

 /**
  * Tests that multiple elements can be rewritten in the matrix 
  * multiple times.
  */
 public void testRewriteMatrix() {
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(10, 10, 1.0);
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(10, 10, 1.0);

  // fill the matrices with simple values
  simpleInitializeMatrix(rMatrix);
  simpleInitializeMatrix(cMatrix);
 
  // check that matrix elements have been set correctly after initial rewrite
  double[][] primitiveMatrix = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
  checkEquivalence(primitiveMatrix, rMatrix);
  checkEquivalence(primitiveMatrix, cMatrix);
  
  // then rewrite elements again
  simpleInitializeMatrix(rMatrix, 2);
  simpleInitializeMatrix(cMatrix, 2);
  
  // check that matrix elements have been set correctly after second rewrite
  double[][] expectedMatrix = {{2,3,4},{5,6,7},{8,9,10},{11,12,13}};
  checkEquivalence(expectedMatrix, rMatrix);
  checkEquivalence(expectedMatrix, cMatrix);
 }
 
 /**
  * Initializes a matrix to a set of simple values. Note that this only
  * initializes the top left-hand 4 columns and 3 rows; all other elements
  * are left unchanged. Top elements of matrix will look like: 
  * counter counter+3 counter+6 counter+9
  * 
  * counter+1 counter+4 counter+7 counter+10
  * 
  * counter+2 counter+5 counter+8 counter+11
  * 
  * Preconditions: matrix size is at least 4 col x 3 rows
  * @param matrix
  */
 private void simpleInitializeMatrix(IDoubleMatrix matrix, int counter) {
  for (int i = 0; i < 4; i++) {
   for (int j = 0; j < 3; j++) {
    try {
     // initialize matrix row-by-row, then column-by-column
     matrix.setEntry(j, i, counter);
     // increment counter so that matrix elements are different, but predictable
     counter++;
    } catch (OutOfBoundsException e) {
     fail("Caught an unexpected OutOfBounds error when doing a valid setEntry.");
    }
   }
  }
 }
 
 /**
  * Initializes a matrix to a set of simple values. Note that this only
  * initializes the top left-hand 4 columns and 3 rows; all other elements
  * are left unchanged. Top elements of matrix will look like: 
  * 1.0 4.0 7.0 10.0
  * 
  * 2.0 5.0 8.0 11.0
  * 
  * 3.0 6.0 9.0 12.0
  * 
  * Precondition: matrix size is at least 4 col x 3 rows
  * @param matrix
  */
 private void simpleInitializeMatrix(IDoubleMatrix matrix) {
  simpleInitializeMatrix(matrix, 1);
 }
 
 /**
  * Checks that all elements of a primitive 2-d array are equivalent to the elements found in a matrix.
  * Note that if the matrix is larger than the primitive 2-d array, the elements in the matrix
  * that do not have any corresponding elements in the 2-d array will not be checked
  * 
  * @param actualMatrix primitive matrix (specified in format of {{col},{col},{col}}
  * @param checkThisMatrix
  */
 private void checkEquivalence(double[][] actualMatrix, IDoubleMatrix checkThisMatrix) {
  // set up for iteration
  int numCols = actualMatrix.length;
  int numRows = actualMatrix[0].length;
  
  // iterate through elements of 2-d array and matrix
  for (int rowIndex = 0; rowIndex < numRows; rowIndex++) {
   for (int colIndex = 0; colIndex < numCols; colIndex++) {

    try {
     double expected = actualMatrix[colIndex][rowIndex];
     double actual = checkThisMatrix.getEntry(rowIndex, colIndex);
     
     // compare corresponding elements from the 2-d array and the IDoubleMatrix
     checkCloseEnough(actual, expected);
    } catch (OutOfBoundsException e) {
     fail("Encountered an out of bounds exception when doing a valid get operation.");
    }
   }
  }
 }
 
 /**
  * Tests if the getRow() function works correctly, also checks for out of bounds access errors.
  */
 public void testSimpleGetRow() {
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(3, 5, 5.0);
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(3, 5, 5.0);
  
  simpleInitializeMatrix(cMatrix);
  simpleInitializeMatrix(rMatrix);
    
  seeIfGetRowWorks(cMatrix, 5.0);
  seeIfGetRowWorks(rMatrix, 5.0);
 }
 
 /**
  * Test if getRow() works as specified. Tests if OutOfBounds exceptions are 
  * thrown correctly with invalid access.
  * 
  * @param matrix
  * @param initialValue the initial value of every element in the matrix
  */
 private void seeIfGetRowWorks(IDoubleMatrix matrix, double initialValue) {
  throwsCorrectExceptionGetRow(matrix, 5);
  throwsCorrectExceptionGetRow(matrix, -1);
  
  try {
   
   checkListsAreEqual(new double[]{1, 4, 7, 10, initialValue}, matrix.getRow(0));
   checkListsAreEqual(new double[]{2, 5, 8, 11, initialValue}, matrix.getRow(1));
   checkListsAreEqual(new double[]{3, 6, 9, 12, initialValue}, matrix.getRow(matrix.getNumRows()-1));
  } catch (OutOfBoundsException e) {
   fail("Went OutOfBounds when doing a valid getRow() operatoin.");
  }
 }
 
 /**
  * Checks that an exception is thrown when an invalid row is accessed in the given matrix
  * with getRow().
  * Precondition: should be called with row index that is out of bounds for the given matrix,
  * as defined in the specification.

  * @param matrix
  * @param rowIndex
  */
 private void throwsCorrectExceptionGetRow(IDoubleMatrix matrix, int rowIndex) {
  boolean errorNotThrown = true;
  // try to get row at an invalid row index; getRow should throw an error
  try {
   matrix.getRow(rowIndex);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  }

  // if no error was detected, then fail the test
  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds exception to be thrown.");
  }
 }
 
 /**
  * Tests if the getColumn() function works correctly, also checks for out of bounds access errors.
  */
 public void testSimpleGetColumn() {
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(4, 4, 5.0);
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(4, 4, 5.0);
  
  simpleInitializeMatrix(cMatrix);
  simpleInitializeMatrix(rMatrix);
  
  seeIfGetColumnWorks(cMatrix, 5.0);
  seeIfGetColumnWorks(rMatrix, 5.0);
 }
 
 /**
  * Test if getColumn() works as specified. Tests if OutOfBounds exceptions are 
  * thrown correctly with invalid access.
  * 
  * @param matrix
  * @param initialValue the initial value of every element in the matrix
  */
 private void seeIfGetColumnWorks(IDoubleMatrix matrix, double initialValue) {
  throwsCorrectExceptionGetColumn(matrix, 5);
  throwsCorrectExceptionGetColumn(matrix, -1);

  try {
   checkListsAreEqual(new double[]{1, 2, 3, initialValue}, matrix.getColumn(0));
   checkListsAreEqual(new double[]{4, 5, 6, initialValue}, matrix.getColumn(1));
   checkListsAreEqual(new double[]{7, 8, 9, initialValue}, matrix.getColumn(2));
  } catch (OutOfBoundsException e) {
   fail("Went OutOfBounds when doing a valid getColumn() operation.");
  }
 }
 
 /**
  * Checks that an exception is thrown when an invalid column is accessed in the given matrix
  * with getColumn().
  * Precondition: should be called with column index that is out of bounds for the given matrix,
  * as defined in the specification.

  * @param matrix
  * @param colIndex
  */
 private void throwsCorrectExceptionGetColumn(IDoubleMatrix matrix, int colIndex) {
  boolean errorNotThrown = true;
  // try to get col at an invalid col index; getColumn should throw an error
  try {
   matrix.getColumn(colIndex);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  }

  // if no error was detected, then fail the test
  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds exception to be thrown.");
  }
 }
 
 /**
  * Tests if the setColumn() function works correctly, also checks for out of bounds access 
  * or illegal argument errors which should be thrown under certain conditions.
  */
 public void testSetColumn() {
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(3, 4, 5.0);
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(3, 4, 5.0);
  
  // setup both matrices
  simpleInitializeMatrix(cMatrix);
  simpleInitializeMatrix(rMatrix);
  
  // test both matrices
  seeIfSetColumnWorks(cMatrix);
  seeIfSetColumnWorks(rMatrix);
 }
 
 /**
  * Test if setColumn() works as specified. Tests if OutOfBounds exceptions are 
  * thrown correctly with invalid access; also tests if IllegalArgument exceptions
  * are thrown when the IDoubleVector argument in setColumn() is null or not of the same
  * length as the columns of the matrix
  * 
  * @param matrix
  */
 private void seeIfSetColumnWorks(IDoubleMatrix matrix) {
  IDoubleVector column = new SparseDoubleVector(3, 9999.0);
  IDoubleVector inappropriateLengthColumn = new SparseDoubleVector(100, -Double.MAX_VALUE);
  IDoubleVector inappropriateLengthColumn2 = new SparseDoubleVector(0, -Double.MAX_VALUE);
  
  // all of these cases should throw an exception - either invalid column indices are to be set, 
  // or the IDoubleVector argument in setColumn is null
  throwsCorrectExceptionSetColumn(matrix, 10, column);
  throwsCorrectExceptionSetColumn(matrix, -1, column);
  throwsCorrectExceptionSetColumn(matrix, 2, inappropriateLengthColumn);
  throwsCorrectExceptionSetColumn(matrix, 2, inappropriateLengthColumn2);

  try {
   // try a valid case, see if column ends up matching the expected column of values
   matrix.setColumn(2, column);
   checkListsAreEqual(new double[] {9999, 9999, 9999}, matrix.getColumn(2));

  } catch (OutOfBoundsException e) {
   fail("Went OutOfBounds when doing a valid setColumn()/getColumn() operation.");
  } 
 }

 /**
  * Tests if OutOfBounds exceptions are thrown correctly with invalid access;
  * also tests if IllegalArgument exceptions are thrown when the
  * IDoubleVector argument in setColumn() is null or not of the same length
  * as the columns of the matrix.
  * 
  * Preconditions: IDoubleVector argument in setColumn() is null or not of the same length
  * as the columns of the matrix or the column index exceeds the bounds of the matrix.
  * 
  * @param matrix
  * @param columnIndex
  * @param column
  */
 private void throwsCorrectExceptionSetColumn(IDoubleMatrix matrix, int columnIndex, IDoubleVector column) {
  boolean errorNotThrown = true;

  // try an invalid setColumn at either an invalid index or with an invalid (null) argument, which should throw an error.
  try {
   matrix.setColumn(columnIndex, column);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  } catch (IllegalArgumentException e) {
   errorNotThrown = false;
  }

  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds or IllegalArgument exception to be thrown.");
  }
 }

 /**
  * Tests if the setRow() function works correctly, also checks for out of bounds access 
  * or illegal argument errors which should be thrown under certain conditions.
  */
 public void testSetRow() {
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(3, 4, 5.0);
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(3, 4, 5.0);

  simpleInitializeMatrix(cMatrix);
  simpleInitializeMatrix(rMatrix);
  
  seeIfSetRowWorks(cMatrix);
  seeIfSetRowWorks(rMatrix);
 }
 
 /**
  * Test if setRow() works as specified. Tests if OutOfBounds exceptions are 
  * thrown correctly with invalid access; also tests if IllegalArgument exceptions
  * are thrown when the IDoubleVector argument in setRow() is null or not of the same
  * length as the columns of the matrix
  * 
  * @param matrix
  */
 private void seeIfSetRowWorks(IDoubleMatrix matrix) {
  IDoubleVector row = new SparseDoubleVector(4, 9999.0);
  IDoubleVector inappropriateLengthRow = new SparseDoubleVector(3, -Double.MAX_VALUE);
  IDoubleVector inappropriateLengthRow2 = new SparseDoubleVector(5, -Double.MAX_VALUE);
  
  // all of these cases should throw an exception - either invalid row indices are to be set, 
  // or the IDoubleVector argument in setRow is null
  throwsCorrectExceptionSetRow(matrix, 10, row);
  throwsCorrectExceptionSetRow(matrix, -1, row);
  throwsCorrectExceptionSetRow(matrix, 2, inappropriateLengthRow);
  throwsCorrectExceptionSetRow(matrix, 2, inappropriateLengthRow2);
  
  // check that the row was set correctly
  try {
   matrix.setRow(2, row);
   checkListsAreEqual(new double[] {9999, 9999, 9999}, matrix.getRow(2));

  } catch (OutOfBoundsException e) {
   fail("Went OutOfBounds when doing a valid setRow()/getRow() operation.");
  }
 }
 
 /**
  * Tests if OutOfBounds exceptions are thrown correctly with invalid access;
  * also tests if IllegalArgument exceptions are thrown when the
  * IDoubleVector argument in setColumn() is null or not of the same length
  * as the columns of the matrix.
  * 
  * Preconditions: IDoubleVector argument in setColumn() is null or not of the same length
  * as the columns of the matrix or the column index exceeds the bounds of the matrix.
  * 
  * @param matrix
  * @param rowIndex
  * @param row
  */
 private void throwsCorrectExceptionSetRow(IDoubleMatrix matrix, int rowIndex, IDoubleVector row) {
  boolean errorNotThrown = true;
  
  // try an invalid setRow at either an invalid index or with an invalid (null) argument, which should throw an error.
  try {
   matrix.setRow(rowIndex, row);
  } catch (OutOfBoundsException e) {
   errorNotThrown = false;
  } catch (IllegalArgumentException e) {
   errorNotThrown = false;
  }

  if (errorNotThrown) {
   fail("Was expecting an OutOfBounds or IllegalArgument exception to be thrown.");
  }
 }
 
 /**
  * Tests a more complex scenario of getting and setting columns and rows.
  */
 public void testMoreComplexGetAndSet() {
  IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix(6, 5, 5.0);
  IDoubleMatrix rMatrix = new RowMajorDoubleMatrix(6, 5, 5.0);

  seeIfComplexSetAndGetWorks(cMatrix);
  seeIfComplexSetAndGetWorks(rMatrix);
 }
 
 /**
  * Tests a more complex scenario of getting and setting columns and rows.
  * Basically, see if overwriting an element in a column by setting a row works, and 
  * see if overwriting an element in a row by setting a column works properly.
  * @param matrix the matrix to be tested
  */
 private void seeIfComplexSetAndGetWorks(IDoubleMatrix matrix) {
  IDoubleVector row = new SparseDoubleVector(5, 100.0);
  
  IDoubleVector column = new SparseDoubleVector(6, -999.0);
  IDoubleVector secondColumn = new SparseDoubleVector(6, 9999.0);
  
  try {
   // also serves as a black-box test: no elements have been initialized at all before
   // the setColumn() and setRow() methods are called
   matrix.setColumn(3, column);
   matrix.setRow(3, row);
   
   checkListsAreEqual(new double[]{-999, -999, -999, 100, -999, -999}, matrix.getColumn(3));
   
   // now set another column, rewriting the value at the intersection of column 3 and row 3
   matrix.setColumn(3, secondColumn);
   checkListsAreEqual(new double[]{100, 100, 100, 9999, 100}, matrix.getRow(3));
   
   // check that getEntry returns the correct result at the intersection of this changed row and column
   checkCloseEnough(matrix.getEntry(3,3), 9999);
   
  } catch (OutOfBoundsException e) {
   fail("Went OutOfBounds on a valid row or column access.");
  }
 }
 
 /**
  * Tests access in boundary cases for matrices of various sizes - 0x0, 0x1, 1x0, and 1x1.
  */
 public void testBoundaryCases() {
  // make 0x0, 0x1, 1x0, and 1x1 row-major matrices
  IDoubleMatrix rMatrixSizeZero = new RowMajorDoubleMatrix(0, 0, 0.0);
  IDoubleMatrix rMatrixSizeOneColumn = new RowMajorDoubleMatrix(0, 1, 0.0); 
  IDoubleMatrix rMatrixSizeOneRow = new RowMajorDoubleMatrix(1, 0, 0.0);
  IDoubleMatrix rMatrixSizeOne = new RowMajorDoubleMatrix(1, 1, 0.0);
  
  // test that any accesses with get entry will produce an OutOfBoundsException
  throwsCorrectExceptionWithGet(rMatrixSizeZero, 0, 0);
  throwsCorrectExceptionWithGet(rMatrixSizeOneColumn, 0, 0);
  throwsCorrectExceptionWithGet(rMatrixSizeOneRow, 0, 0);
  throwsCorrectExceptionWithGet(rMatrixSizeOne, 1, 1);
  
  // make 0x0, 0x1, 1x0, and 1x1 column-major matrices
  IDoubleMatrix cMatrixSizeZero = new ColumnMajorDoubleMatrix(0, 0, 0.0);
  IDoubleMatrix cMatrixSizeOneColumn = new ColumnMajorDoubleMatrix(0, 1, 0.0);
  IDoubleMatrix cMatrixSizeOneRow = new ColumnMajorDoubleMatrix(1, 0, 0.0);
  IDoubleMatrix cMatrixSizeOne = new ColumnMajorDoubleMatrix(1, 1, 0.0);
  
  // test that any accesses with get entry will produce an OutOfBoundsException
  throwsCorrectExceptionWithGet(cMatrixSizeZero, 0, 0);
  throwsCorrectExceptionWithGet(cMatrixSizeOneColumn, 0, 0);
  throwsCorrectExceptionWithGet(cMatrixSizeOneRow, 0, 0);
  throwsCorrectExceptionWithGet(cMatrixSizeOne, 1, 1);
 }
 
 
  /**
   * Tests the matrix by writing few entries in a really large matrix.
   * The dimensions of the matrix should preferably be atleast 500x500.
   */
  private void sparseMatrixLargeTester (IDoubleMatrix checkMe) {
    // Get dimensions
    int numColumns = checkMe.getNumColumns();
    int numRows = checkMe.getNumRows();
    
    // Reset individual entries
    for (int i = 0; i < numColumns; i+=100) {
      for (int j = 0; j < numRows; j+=100) {
        try {
          double entry = (j+1)*10+(i+1);
          checkMe.setEntry(j,i,entry);
        } catch (OutOfBoundsException e) {
          fail("Was unable to set the entry column: " + i + " row: " + j);
        }
      }
    }

    // Get individual entries
    for (int i = 0; i < numColumns; i+=100) {
      for (int j = 0; j < numRows; j+=100) {
        try {
          double entry = (j+1)*10+(i+1);
          assertEquals(entry, checkMe.getEntry(j,i));
        } catch (OutOfBoundsException e) {
          fail("Was unable to get the entry column: " + i + " row: " + j);
        }
      }
    }
  }
  
  /**
   * Tests the matrix by writing around 1% of the matrix in a sparse manner for a RowMajorDoubleMatrix
   */
  public void testSparseRowMajor () {
    System.out.println("Testing sparse RowMajorDoubleMatrix.");
    IDoubleMatrix rMatrix = new RowMajorDoubleMatrix (1000,1000,-1.0);
    sparseMatrixLargeTester(rMatrix);
  }
  
  /**
   * Tests the matrix by writing around 1% of the matrix in a sparse manner for a RowMajorDoubleMatrix
   */
  public void testSparseColumnMajor () {
    System.out.println("Testing sparse ColumnMajorDoubleMatrix.");
    IDoubleMatrix cMatrix = new ColumnMajorDoubleMatrix (1000,1000,-1.0);
    sparseMatrixLargeTester(cMatrix);
  }
  
  
  /**
   * This test runs a test on the following combinations on a wide variety of matrix sizes and indices.  
   * Here, we test 
   *   setting an entry, then setting a row.
   *   setting an entry, then setting a column.
   *   setting a row, then setting an entry.
   *   setting a column, then setting an entry.
   */
  public void testEntryCombos () {
    int numRows;
    int numCols;
    // Change rows represents the row indices that we change to another value (distinct from the initial value)
    int[] changeRows = {0, 0, 0};
    // Change columns represents the row indices that we change to another value (distinct from the initial value)
    int[] changeCols = {0, 0, 0};
    double initVal = 0.0;
    IDoubleMatrix myMatrix;
    SparseDoubleVector setRow;
    SparseDoubleVector setCol;
    
    // Test for all size of matrices (the sizes we mentioned above)
    for (int i = 1; i < 1000; i = i * 10) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          numRows = i + j * 100;
          numCols = i + k * 100;
          
          // We test a middle row index (middle case)
          changeRows[1] = numRows / 2;
          // We also test the last row index (boundary case)
          changeRows[2] = numRows - 1;
          
          // We test a middle column index (middle case)
          changeCols[1] = numCols / 2;
          // We also test the last column index (boundary case)
          changeCols[2] = numCols - 1;
          
          // We use values from setValues due to the reason we mention at the top.  
          // Here, we test setting a row, then an entry (or the other way around) on all possible combinations from 
          // a corner to a boundary, a boundary to a corner, or even middle cases.  We also verify that when they don't
          // intersect, that it works as well.
          for (double testVal : setValues) {
            for (double changeVal : setValues) {
              // Iterate through middle, top, and bottom rows
              for (int changeRow : changeRows) {
                // Iterate through middle and corner entries.
                for (int change_i_Entry : changeCols) {
                  for (int change_j_Entry : changeRows) {
                    // Test setting an entry and then the row on a column major double matrix
                    myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                    setRow = new SparseDoubleVector(numCols, testVal);
                    testSetEntryRowCombo(myMatrix, change_i_Entry, change_j_Entry, changeRow, changeVal, setRow);
                    
                    // Test setting a row and then the entry on a column major double matrix
                    myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                    setRow = new SparseDoubleVector(numCols, testVal);
                    testSetRowEntryCombo(myMatrix, change_i_Entry, change_j_Entry, changeRow, changeVal, setRow);
                    
                    // Test setting an entry and then the row on a row major double matrix
                    myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                    setRow = new SparseDoubleVector(numCols, testVal);
                    testSetEntryRowCombo(myMatrix, change_i_Entry, change_j_Entry, changeRow, changeVal, setRow);
                    
                    // Test setting a row and then the entry on a row major double matrix
                    myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                    setRow = new SparseDoubleVector(numCols, testVal);
                    testSetRowEntryCombo(myMatrix, change_i_Entry, change_j_Entry, changeRow, changeVal, setRow);
                  }
                }
              }
            }
          }
          
          // We use values from setValues due to the reason we mention at the top.  
          // Here, we test setting a column, then an entry (or the other way around) on all possible combinations from 
          // a corner to a boundary, a boundary to a corner, or even middle cases.  We also verify that when they don't
          // intersect, that it works as well.
          for (double testVal : setValues) {
            for (double changeVal : setValues) {
              // Iterate through middle, leftmost, and rightmost columns
              for (int changeCol : changeCols) {
                // Iterate through middle and corner entries.
                for (int change_i_Entry : changeCols) {
                  for (int change_j_Entry : changeRows) {
                    // Test setting an entry and then the column on a column major double matrix
                    myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                    setCol = new SparseDoubleVector(numRows, testVal);
                    testSetEntryColCombo(myMatrix, change_i_Entry, change_j_Entry, changeCol, changeVal, setCol);
                    
                    // Test setting a column and then the entry on a column major double matrix
                    myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                    setCol = new SparseDoubleVector(numRows, testVal);
                    testSetColEntryCombo(myMatrix, change_i_Entry, change_j_Entry, changeCol, changeVal, setCol);
                    
                    // Test setting an entry and then the column on a row major double matrix
                    myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                    setCol = new SparseDoubleVector(numRows, testVal);
                    testSetEntryColCombo(myMatrix, change_i_Entry, change_j_Entry, changeCol, changeVal, setCol);
                    
                    // Test setting a column and then the entry on a row major double matrix
                    myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                    setCol = new SparseDoubleVector(numRows, testVal);
                    testSetColEntryCombo(myMatrix, change_i_Entry, change_j_Entry, changeCol, changeVal, setCol);
                  }
                }
              }
            }
          }
        }
      }
    }
    
  }
  
  /**
   * This test runs a test on the following combinations on a wide variety of matrix sizes and indices.  
   * Here, we test 
   *   setting a column, then setting a row.
   *   setting a row, then setting a column.
   */
  public void testColRowCombos () {
    int numRows;
    int numCols;
    // Change rows represents the row indices that we change to another value (distinct from the initial value)
    int[] changeRows = {0, 0, 0};
    // Change columns represents the row indices that we change to another value (distinct from the initial value)
    int[] changeCols = {0, 0, 0};
    double initVal = 0.0;
    IDoubleMatrix myMatrix;
    SparseDoubleVector setRow;
    SparseDoubleVector setCol;
    
    // Test for all size of matrices as mentioed above.
    for (int i = 1; i < 1000; i = i * 1000) {
      for (int j = 0; j < 3; j=j+2) {
        for (int k = 0; k < 3; k=k+2) {
          numRows = i + j * 100;
          numCols = i + k * 100;
          
          // We test a middle row index (middle case)
          changeRows[1] = numRows / 2;
          // We also test the last row index (boundary case)
          changeRows[2] = numRows - 1;
          
          // We test a middle column index (middle case)
          changeCols[1] = numCols / 2;
          // We also test the last column index (boundary case)
          changeCols[2] = numCols - 1;
          
          // We use values from setValues due to the reason we mention at the top.  
          // Here, we test setting a column, then a row (or the other way around) on all possible combinations from 
          // a boundary to a boundary, or a boundary to a middle, to even middle to middle.  We also verify that when they don't
          // intersect, that it works as well.
          for (double testVal : setValues) {
            for (double changeVal : setValues) {
              // Iterate through middle, top, and bottom rows
              for (int changeRow : changeRows) {
                // Iterate through middle, leftmost, and rightmost columns
                for (int changeCol : changeCols) { 
                  // Test setting a row and then a column on a column major double matrix
                  myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                  setRow = new SparseDoubleVector(numCols, testVal);
                  setCol = new SparseDoubleVector(numRows, changeVal);
                  testRowColCombo(myMatrix, changeCol, changeRow, setRow, setCol);
                  // Test setting a column and then a row on a column major double matrix
                  myMatrix = new ColumnMajorDoubleMatrix(numRows, numCols, 0.0);
                  setRow = new SparseDoubleVector(numCols, testVal);
                  setCol = new SparseDoubleVector(numRows, changeVal);
                  testColRowCombo(myMatrix, changeCol, changeRow, setRow, setCol);
                  // Test setting a row and then the column on a row major double matrix
                  myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                  setRow = new SparseDoubleVector(numCols, testVal);
                  setCol = new SparseDoubleVector(numRows, changeVal);
                  testRowColCombo(myMatrix, changeCol, changeRow, setRow, setCol);
                  // Test setting a column and then a row on a row major double matrix
                  myMatrix = new RowMajorDoubleMatrix(numRows, numCols, 0.0);
                  setRow = new SparseDoubleVector(numCols, testVal);
                  setCol = new SparseDoubleVector(numRows, changeVal);
                  testColRowCombo(myMatrix, changeCol, changeRow, setRow, setCol);
                }
              }
            }
          }
        }
      }
    }
  }
  
  
  /**
   * Here, we test first setting an entry in the matrix, then setting a row within the matrix.  
   * We check that the row is correct as well as the entry we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_entry      The column for which we set the entry
   * @param j_entry      The row for which we set the entry
   * @param j_row        The index of the row that we will set
   * @param entry_val    The value that we will setValue with
   * @param row_val      the row that we will setRow with
   */
  private void testSetEntryRowCombo (IDoubleMatrix myMatrix, int i_entry, int j_entry, int j_row, double entry_val, IDoubleVector row_val) {
    // An IDoubleVector that contains the correct row
    IDoubleVector correct_row;
    // A double that keeps track of what the correct value is.
    double correct_val;
    
    // We wrap it in a try statement since if we catch an OutOfBoundsException, then something went wrong.
    try {
      // Set the correct value to be initially the entry value.
      correct_val = entry_val;
      // Note though that since we change the row later, if the row that we change contains the index of the
      // entry value that we will set, then the correct value will be changed to what is contained in the row. 
      // We account for that here by changing the correct value if that happens.
      if (j_entry == j_row) {
        correct_val = row_val.getItem(i_entry);
      }
      // The correct row is the row passed in since the row will never change.
      correct_row = row_val;
      
      // We set the entry and row here
      myMatrix.setEntry(j_entry, i_entry, entry_val);
      myMatrix.setRow(j_row, row_val);
      // We then compare if it matches with what was expected.
      testGetRow(myMatrix, j_row, correct_row);
      testGetEntry(myMatrix, i_entry, j_entry, correct_val);
    } catch (OutOfBoundsException e) {
      fail(e.getMessage());
    }
  }
  
  /**
   * Here, we test first setting a row in the matrix, then setting an entry within the matrix.  
   * We check that the row is correct as well as the entry we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_entry      The column for which we set the entry
   * @param j_entry      The row for which we set the entry
   * @param j_row        The index of the row that we will set
   * @param entry_val    The value that we will setValue with
   * @param row_val      the row that we will setRow with
   */
  private void testSetRowEntryCombo (IDoubleMatrix myMatrix, int i_entry, int j_entry, int j_row, double entry_val, IDoubleVector row_val) {
    // An IDoubleVector that contains the correct row
    IDoubleVector correct_row;
    // A double that keeps track of what the correct value is.
    double correct_val;
    try {
      // Set the correct value to the entry value.
      correct_val = entry_val;
      correct_row = row_val;
      
      // We set the row, then the entry here
      myMatrix.setRow(j_row, row_val);
      myMatrix.setEntry(j_entry, i_entry, entry_val);
      
      // Note though that if the entry we set resides on the row, then that will 
      // change the value contained in the row.  We change the correct_row to reflect such.
      if (j_entry == j_row) {
        correct_row.setItem(i_entry, entry_val);
      }
      
      // Now, we test that the values are indeed correct.
      testGetRow(myMatrix, j_row, row_val);
      testGetEntry(myMatrix, i_entry, j_entry, correct_val);
    } catch (OutOfBoundsException e) {
      fail("setEntry setRow combo Failed: Encountered Out of Bounds Exception");
    }
  }
  
  /**
   * Here, we test first setting an entry in the matrix, then setting a column within the matrix.  
   * We check that the column is correct as well as the entry we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_entry      The column for which we set the entry
   * @param j_entry      The row for which we set the entry
   * @param i_col        The index of the row that we will set
   * @param entry_val    The value that we will setValue with
   * @param col_val      The column that we will setColumn with
   */
  private void testSetEntryColCombo (IDoubleMatrix myMatrix, int i_entry, int j_entry, int i_col, double entry_val, IDoubleVector col_val) {
    IDoubleVector correct_col;
    // A double that keeps track of what the correct value is.
    double correct_val;
    try {
      // Set the correct value to the entry value.
      correct_val = entry_val;
      
      // Note though that since we change the column later, if the column that we change contains the index of the
      // entry value that we will set, then the correct value will be changed to what is contained in the column. 
      // We account for that here by changing the correct value if that happens.
      if (i_entry == i_col) {
        correct_val = col_val.getItem(j_entry);
      }
      
      // The correct column is the column passed in since the column will never change.
      correct_col = col_val;
      
      // We set the entry and the column here.
      myMatrix.setEntry(j_entry, i_entry, entry_val);
      myMatrix.setColumn(i_col, col_val);
      
      // Now, we test that the values are indeed correct.
      testGetColumn(myMatrix, i_col, correct_col);
      testGetEntry(myMatrix, i_entry, j_entry, correct_val);
    } catch (OutOfBoundsException e) {
      fail(e.getMessage());
    }
  }
  
  /**
   * Tests first setting a column in the matrix, then setting an entry within the matrix.  
   * We check that the column is correct as well as the entry we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_entry      The column for which we set the entry
   * @param j_entry      The row for which we set the entry
   * @param i_col        The index of the row that we will set
   * @param row_val      The row that we will setRow with
   * @param col_val      The column that we will setColumn with
   */
  private void testSetColEntryCombo (IDoubleMatrix myMatrix, int i_entry, int j_entry, int i_col, double entry_val, IDoubleVector col_val) {
    IDoubleVector correct_col;
    // A double that keeps track of what the correct value is.
    double correct_val;
    try {
      // Set the correct value to the entry value.
      correct_val = entry_val;
      correct_col = col_val;
      
      // We set the entry and the column here.
      myMatrix.setColumn(i_col, col_val);
      myMatrix.setEntry(j_entry, i_entry, entry_val);
      
      // Note though that if the entry we set resides on the column, then that will 
      // change the value contained in the column.  We change the correct_col to reflect such.
      if (i_entry == i_col) {
        correct_col.setItem(j_entry, entry_val);
      }
      
      // Now, we test that the values are indeed correct.
      testGetColumn(myMatrix, i_col, col_val);
      testGetEntry(myMatrix, i_entry, j_entry, correct_val);
    } catch (OutOfBoundsException e) {
      fail("setEntry setRow combo Failed: Encountered Out of Bounds Exception");
    }
  }
  
  /**
   * Tests first setting a column in the matrix, then setting a row within the matrix.  
   * We check that the column is correct as well as the row we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_col        The index of the row that we will set
   * @param j_row        The index of the column that we will set
   * @param row_val      The row that we will setRow with
   * @param col_val      the column that we will setColumn with
   */
  private void testRowColCombo (IDoubleMatrix myMatrix, int i_col, int j_row, IDoubleVector row_val, IDoubleVector col_val) {
    // IDoubleVectors that hold the correct column and the correct row
    IDoubleVector correct_col;
    IDoubleVector correct_row;
    
    try {
      // The correct row is set to the passed in column and won't change since we change the row after we change the column.
      correct_row = row_val;
      // The correct column is set to the passed in column (will change since row and column will always overlap)
      correct_col = col_val;
      
      // Now, we set the column and the row to the column/row passed in to the desired indices.
      myMatrix.setColumn(i_col, col_val);
      myMatrix.setRow(j_row, row_val);
      
      // We now change the corresponding item in the column that would've been changed.
      correct_col.setItem(j_row, row_val.getItem(0));
      
      // We verify here that the column and row match what should be the correct column and row.
      testGetColumn(myMatrix, i_col, correct_col);
      testGetRow(myMatrix, j_row, correct_row);
    } catch (OutOfBoundsException e) {
      fail("setRow setCol combo Failed: Encountered Out of Bounds Exception");
    }
  }
  
  
  /**
   * Tests first setting a column in the matrix, then setting a row within the matrix.  
   * We check that the column is correct as well as the row we changed within the matrix.
   * 
   * @param myMatrix     The matrix that is passed in
   * @param i_col        The index of the row that we will set
   * @param j_row        The index of the column that we will set
   * @param row_val      The row that we will setRow with
   * @param col_val      the column that we will setColumn with
   */
  private void testColRowCombo (IDoubleMatrix myMatrix, int i_col, int j_row, IDoubleVector row_val, IDoubleVector col_val) {
    // IDoubleVectors that hold the correct column and the correct row
    IDoubleVector correct_col;
    IDoubleVector correct_row;
    
    try {
      // The correct row is set to the passed in column initially (will change since row and column will always overlap)
      correct_row = row_val;
      // The correct column is set to the passed in column and won't change since we change the column after we change the row.
      correct_col = col_val;
      
      // Now, we set the column and the row to the column/row passed in to the desired indices.
      myMatrix.setRow(j_row, row_val);
      myMatrix.setColumn(i_col, col_val);
      
      // We now change the corresponding item in the row that would've been changed.
      correct_row.setItem(i_col, col_val.getItem(0));
      
      // We verify here that the column and row match what should be the correct column and row.
      testGetColumn(myMatrix, i_col, correct_col);
      testGetRow(myMatrix, j_row, correct_row);
    } catch (OutOfBoundsException e) {
      fail("setRow setCol combo Failed: Encountered Out of Bounds Exception");
    }
  }
  
  /**
   * Tests that getRow works.  We do so by comparing the row retrieved by getRow with
   * an IDoubleVector that represents the row we expect to get.
   * 
   * @param myMatrix       Matrix that is tested
   * @param j              Index of row that we want to retrieve
   * @param expectedRow    Expected row that we compare with our retrieved j'th row.
   */
  private void testGetRow (IDoubleMatrix myMatrix, int j, IDoubleVector expectedRow) {
    IDoubleVector retrievedRow;
    
    try {
      // Retrieve the row from myMatrix
      retrievedRow = myMatrix.getRow(j);
      // In order to test that the retrieved row is correct, we iterate through every element in the retrieved row and compare it
      // with the expected row
      for (int i = 0; i < myMatrix.getNumColumns(); i++) {
        assertTrue("getRow() Failed: Retrieved row did not match expected row", retrievedRow.getItem(i) == expectedRow.getItem(i));
      }
    } catch (OutOfBoundsException e) {
      // We fail instantly if we ever encounter an OutOfBoundsException
      fail("getRow() Failed: Encountered OutOfBoundsException");
    }
  }
  
  /**
   * Tests that getEntry works.  We do so by utilizing getEntry, then verifying that
   * the entry retrieved matches what we expected.
   * 
   * @param myMatrix       Matrix that is tested
   * @param i              Column that we retrieve from
   * @param j              Row that we retrieve from
   * @param expectedVal    Expected value that we expect to get from the i, j'th entry.
   */
  private void testGetEntry (IDoubleMatrix myMatrix, int i, int j, double expectedVal) {
    // Attempt to retrieve entry.  If it does not match, we print an appropriate error.  If we encounter an OutOfBoundsException, we print a different error.
    try {
      assertTrue("getEntry() Failed: Retrieved value did not match expected value", myMatrix.getEntry(j, i) == expectedVal);
    } catch (OutOfBoundsException e) {
      fail("getEntry() Failed: Encountered OutOfBoundsException");
    }
  }
  
   /**
   * Tests that getColumn works.  We do so by comparing the column retrieved by getColumn with
   * an IDoubleVector that represents the column we expect to get.
   * 
   * @param myMatrix       Matrix that is tested
   * @param i              Index of column that we want to retrieve
   * @param expectedCol    Expected column that we compare with our retrieved i'th column.
   */
  private void testGetColumn (IDoubleMatrix myMatrix, int i, IDoubleVector expectedCol) {
    IDoubleVector retrievedColumn;
    try {
      // Retrieve the column from myMatrix
      retrievedColumn = myMatrix.getColumn(i);
      // In order to test that the retrieved column is correct, we iterate through every element in the retrieved column and compare it
      // with the expected column
      for (int j = 0; j < myMatrix.getNumRows(); j++) { 
        assertTrue("getColumn() Failed: Retrieved column did not match expected column", retrievedColumn.getItem(j) == expectedCol.getItem(j));
      } 
    } catch (OutOfBoundsException e) {
      // We fail instantly if we ever encounter an OutOfBoundsException
      fail("getColumn() Failed: Encountered OutOfBoundsException");
    }
  }
  
  // create and return a new random vector of the specified length
  private IDoubleVector createRandomVector (int len, Random useMe) {
    IDoubleVector returnVal = new DenseDoubleVector (len, 0.0);
    for (int i = 0; i < len; i++) {
      try {
        returnVal.setItem (i, useMe.nextDouble ());
      } catch (Exception e) {
        throw new RuntimeException ("bad test case"); 
      }
    }
    return returnVal;
  }
  
  // fills a double matrix, leaveing some rows empty... returns an array with all of the vals
  private double [][] fillMatrixLeaveEmptyRows (IDoubleMatrix fillMe, Random useMe) {
    
    double [][] returnVal = new double[fillMe.getNumRows ()][fillMe.getNumColumns ()];
    
    // fill up all of the rows
    for (int i = 0; i < fillMe.getNumRows (); i++) {
      
      // skip every other one
      try {
        if (useMe.nextInt (2) == 1) {
          IDoubleVector temp = createRandomVector (fillMe.getNumColumns (), useMe);
          fillMe.setRow (i, temp);
          for (int j = 0; j < fillMe.getNumColumns (); j++) {
            returnVal[i][j] = temp.getItem (j); 
          }
        }
      } catch (Exception e) {
        fail ("Died when trying to fill the matrix.");
      }
    }
    
    return returnVal;
  }
  
    // fills a double matrix, leaveing some rows empty... returns an array with all of the vals
  private double [][] fillMatrixLeaveEmptyColumns (IDoubleMatrix fillMe, Random useMe) {
    
    double [][] returnVal = new double[fillMe.getNumRows ()][fillMe.getNumColumns ()];
    
    // fill up all of the columns
    for (int i = 0; i < fillMe.getNumColumns (); i++) {
      
      // skip every other one
      try {
        if (useMe.nextInt (2) == 1) {
          IDoubleVector temp = createRandomVector (fillMe.getNumRows (), useMe);
          fillMe.setColumn (i, temp);
          for (int j = 0; j < fillMe.getNumRows (); j++) {
            returnVal[j][i] = temp.getItem (j); 
          }
        }
      } catch (Exception e) {
        fail ("Died when trying to fill the matrix.");
      }
      
    }
    
    return returnVal;
  }
  
  // fills a double matrix, leaveing some rows empty... returns an array with all of the vals
  private double [][] fillMatrixLeaveEmptyCols (IDoubleMatrix fillMe, Random useMe) {
    
    double [][] returnVal = new double[fillMe.getNumRows ()][fillMe.getNumColumns ()];
    
    // fill up all of the columns
    for (int i = 0; i < fillMe.getNumColumns (); i++) {
      
      // skip every other one
      try {
        if (useMe.nextInt (2) == 1) {
          IDoubleVector temp = createRandomVector (fillMe.getNumRows (), useMe);
          fillMe.setColumn (i, temp);
          for (int j = 0; j < fillMe.getNumRows (); j++) {
            returnVal[j][i] = temp.getItem (j); 
          }
        }
      } catch (Exception e) {
        fail ("Died when trying to fill the matrix.");
      }
      
    }
    
    return returnVal;
  }
  
  private void makeSureAdditionResultIsCorrect (IDoubleMatrix first, IDoubleMatrix second,
                                                  double [][] firstArray, double [][] secondArray) {
    
    // go through and make sure that second has the sum, and that first and firstArray are the same
    try {
      for (int i = 0; i < first.getNumRows (); i++) {
        for (int j = 0; j < second.getNumColumns (); j++) {
          assertTrue (firstArray[i][j] + secondArray[i][j] == second.getEntry (i, j));
          assertTrue (firstArray[i][j] == first.getEntry (i, j));
        }
      }
    } catch (Exception e) {
      fail ("Died when trying to check the matrix.");
    }
  }
  
  private void makeSureSumColumnsIsCorrect (IDoubleVector res, int numRows, int numColumns, double [][] firstArray) {
    
    // go through and make sure that each row is right
    try {
      for (int i = 0; i < numRows; i++) {
        double total = 0.0;
        for (int j = 0; j < numColumns; j++) {
          total += firstArray[i][j];
        }
        assertTrue (total == res.getItem (i));
      }
    } catch (Exception e) {
      fail ("Died when trying to check the res.");
    }
  }
    
    
  private void makeSureSumRowsIsCorrect (IDoubleVector res, int numRows, int numColumns, double [][] firstArray) {
    
    // go through and make sure that each row is right
    try {
      for (int i = 0; i < numColumns; i++) {
        double total = 0.0;
        for (int j = 0; j < numRows; j++) {
          total += firstArray[j][i];
        }
        assertTrue (total == res.getItem (i));
      }
    } catch (Exception e) {
      fail ("Died when trying to check the res.");
    }
  }
  
  public void testAddRowMajorToRowMajor () {
    try {
      // create two different row major matrices
      Random myRand = new Random (122);
      IDoubleMatrix first = new RowMajorDoubleMatrix (200, 100, 0.0);
      IDoubleMatrix second = new RowMajorDoubleMatrix (200, 100, 0.0);
      double [][] firstArray = fillMatrixLeaveEmptyRows (first, myRand);
      double [][] secondArray = fillMatrixLeaveEmptyRows (second, myRand);
      
      // add them together
      first.addMyselfToHim (second);
      
      // and make sue we got the right answer
      makeSureAdditionResultIsCorrect (first, second, firstArray, secondArray);
    } catch (Exception e) {
      fail ("Died when trying to fill the matrix.");
    }
  }
  
  public void testAddColumnMajorToColumnMajor () {
    try {
      // create two different row major matrices
      Random myRand = new Random (122);
      IDoubleMatrix first = new ColumnMajorDoubleMatrix (200, 100, 0.0);
      IDoubleMatrix second = new ColumnMajorDoubleMatrix (200, 100, 0.0);
      double [][] firstArray = fillMatrixLeaveEmptyColumns (first, myRand);
      double [][] secondArray = fillMatrixLeaveEmptyColumns (second, myRand);
      
      // add them together
      first.addMyselfToHim (second);
      
      // and make sue we got the right answer
      makeSureAdditionResultIsCorrect (first, second, firstArray, secondArray);
    } catch (Exception e) {
      fail ("Died when trying to fill the matrix.");
    }
  }
  
  public void testAddColumnMajorToRowMajor () {
    try {
      // create two different row major matrices
      Random myRand = new Random (122);
      IDoubleMatrix first = new ColumnMajorDoubleMatrix (200, 100, 0.0);
      IDoubleMatrix second = new RowMajorDoubleMatrix (200, 100, 0.0);
      double [][] firstArray = fillMatrixLeaveEmptyColumns (first, myRand);
      double [][] secondArray = fillMatrixLeaveEmptyRows (second, myRand);
      
      // add them together
      first.addMyselfToHim (second);
      
      // and make sue we got the right answer
      makeSureAdditionResultIsCorrect (first, second, firstArray, secondArray);
    } catch (Exception e) {
      fail ("Died when trying to fill the matrix.");
    }
  }
  
  public void testAddRowMajorToColumnMajor () {
    try {
      // create two different row major matrices
      Random myRand = new Random (122);
      IDoubleMatrix first = new RowMajorDoubleMatrix (200, 100, 0.0);
      IDoubleMatrix second = new ColumnMajorDoubleMatrix (200, 100, 0.0);
      double [][] firstArray = fillMatrixLeaveEmptyRows (first, myRand);
      double [][] secondArray = fillMatrixLeaveEmptyColumns (second, myRand);
      
      // add them together
      first.addMyselfToHim (second);
      
      // and make sue we got the right answer
      makeSureAdditionResultIsCorrect (first, second, firstArray, secondArray);
    } catch (Exception e) {
      fail ("Died when trying to fill the matrix.");
    }
  }
    
  public void testSumColumnsInColumnMatrix () {
    Random myRand = new Random (122);
    IDoubleMatrix tester = new ColumnMajorDoubleMatrix (200, 200, 0.0);
    double [][] array = fillMatrixLeaveEmptyColumns (tester, myRand);
    IDoubleVector res = tester.sumColumns ();
    makeSureSumColumnsIsCorrect (res, 200, 200, array);
  }
  
  public void testSumColumnsInRowMatrix () {
    Random myRand = new Random (122);
    IDoubleMatrix tester = new RowMajorDoubleMatrix (150, 150, 0.0);
    double [][] array = fillMatrixLeaveEmptyRows (tester, myRand);
    IDoubleVector res = tester.sumColumns ();
    makeSureSumColumnsIsCorrect (res, 150, 150, array);
  }
  
  public void testSumRowsInColumnMatrix () {
    Random myRand = new Random (122);
    IDoubleMatrix tester = new ColumnMajorDoubleMatrix (200, 100, 0.0);
    double [][] array = fillMatrixLeaveEmptyColumns (tester, myRand);
    IDoubleVector res = tester.sumRows ();
    makeSureSumRowsIsCorrect (res, 200, 100, array);
  }
  
  public void testSumRowsInRowMatrix () {
    Random myRand = new Random (122);
    IDoubleMatrix tester = new RowMajorDoubleMatrix (100, 200, 0.0);
    double [][] array = fillMatrixLeaveEmptyRows (tester, myRand);
    IDoubleVector res = tester.sumRows ();
    makeSureSumRowsIsCorrect (res, 100, 200, array);
  }
  
  
  
}
