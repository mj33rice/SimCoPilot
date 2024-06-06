import junit.framework.TestCase;
import java.util.Calendar;
import java.util.*;


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

class LinearIndexedIterator<T> implements Iterator<IIndexedData<T>> {
    private int curIdx;
    private int numElements;
    private int [] indices;
    private Vector<T> data;

    public LinearIndexedIterator (int [] indices, Vector<T> data, int numElements) {
        this.curIdx = -1;
        this.numElements = numElements;
        this.indices = indices;
        this.data = data;
    }

    public boolean hasNext() {
        return (curIdx+1) < numElements;
    }

    public IndexedData<T> next() {
        curIdx++;
        return new IndexedData<T> (indices[curIdx], data.get(curIdx));
    }
    
    public void remove() throws UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
}

class LinearSparseArray<T> implements ISparseArray<T> {
  private int [] indices;
  private Vector <T> data;
  private int numElements;
  private int lastSlot;
  private int lastSlotReturned;

  private void doubleCapacity () {
   data.ensureCapacity (lastSlot * 2);
   int [] temp = new int [lastSlot * 2];
   for (int i = 0; i < numElements; i++) {
     temp[i] = indices[i];  
   }
   indices = temp;
   lastSlot *= 2;
  }
  
  public LinearIndexedIterator<T> iterator () {
      return new LinearIndexedIterator<T> (indices, data, numElements);
  }
   
  public LinearSparseArray(int initialSize) {
      indices = new int[initialSize];
      data = new Vector<T> (initialSize);
      numElements = 0;
      lastSlotReturned = -1;
      lastSlot = initialSize;
  }

  public void put (int position, T element) {
    
    // first see if it is OK to just append
    if (numElements == 0 || position > indices[numElements - 1]) {
      data.add (element);
      indices[numElements] = position;
    
    // if the one ew are adding is not the largest, can't just append
    } else {

      // first check to see if we have data at that position
      for (int i = 0; i < numElements; i++) {
        if (indices[i] == position) {
          data.setElementAt (element, i);
          return;
        }
      }
      
      // if we made it here, there is no data at the position
      int pos;
      for (pos = numElements; pos > 0 && indices[pos - 1] > position; pos--) {
        indices[pos] = indices[pos - 1];
      }
      indices[pos] = position;
      data.add (pos, element);
    }
    
    numElements++;
    if (numElements == lastSlot) 
      doubleCapacity ();
  }

  public T get (int position) {

    // first we find an index value that is less than or equal to the position we want
    for (; lastSlotReturned >= 0 && indices[lastSlotReturned] >= position; lastSlotReturned--);
  
    // now we go up, and find the first one that is bigger than what we want
    for (; lastSlotReturned + 1 < numElements && indices[lastSlotReturned + 1] <= position; lastSlotReturned++);
  
    // now there are three cases... if we are at position -1, then we have a very small guy
    if (lastSlotReturned == -1)
      return null;
  
    // if the guy where we are is less than what we want, we didn't find it
    if (indices[lastSlotReturned] != position)
      return null;
  
    // otherwise, we got it!
    return data.get(lastSlotReturned);
  }
}

class TreeSparseArray<T> implements ISparseArray<T> {
  private TreeMap<Integer, T> array;

  public TreeSparseArray() {
      array = new TreeMap<Integer, T>();
  }

  public void put (int position, T element) {
      array.put(position, element);
  }

  public T get (int position) {
      return array.get(position);
  }

  public IndexedIterator<T> iterator () {
      return new IndexedIterator<T> (array);
  }
}

/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
public class SparseArrayTester extends TestCase {
  
  /**
   * Checks to see if an observed value is close enough to the expected value
   */
  private void checkCloseEnough (double observed, double goal, int pos) { 
    
    // first compute the allowed error 
    double allowedError = goal * 10e-5; 
    if (allowedError < 10e-5) 
      allowedError = 10e-5; 
    
     // if it is too large, then fail 
    if (!(observed > goal - allowedError && observed < goal + allowedError)) 
      if (pos != -1) 
        fail ("Got " + observed + " at pos " + pos + ", expected " + goal); 
      else 
        fail ("Got " + observed + ", expected " + goal); 
  } 
    
  /**
   * This does a bunch of inserts to the array that is being tested, then
   * it does a scan through the array.  The duration of the test in msec is
   * returned to the caller
   */
  private long doInsertThenScan (ISparseArray <Double> myArray) {
    
    // This test tries to simulate how a spare array might be used 
    // when it is embedded inside of a sparse double vector.  Someone first
    // adds a bunch of data into specific slots in the sparse double vector, and
    // then they scan through the sparse double vector from start to finish.  This
    // has the effect of generating a bunch of calls to "get" in the underlying 
    // sparse array, in sequence.  
    
    // get the time
    long msec = Calendar.getInstance().getTimeInMillis();
    
    // add a bunch of stuff into it
    System.out.println ("Doing the inserts...");
    for (int i = 0; i < 1000000; i++) {
      if (i % 100000 == 0)
        System.out.format ("%d%% done ", i * 10 / 100000);
      
      // put one item at an even position
      myArray.put (i * 10, i * 1.0);
      
      // and put another one at an odd position
      myArray.put (i * 10 + 3, i * 1.1);
    }
    
    // now, iterate through the list
    System.out.println ("\nDoing the lookups...");
    double total = 0.0;
    
    // note that we are only going to look at the data in the even positions
    for (int i = 0; i < 20000000; i += 2) {
      if (i % 2000000 == 0)
        System.out.format ("%d%% done ", i * 10 / 2000000);
      Double temp = myArray.get (i);
      
      // if we got back a null, there was no data there
      if (temp != null)
        total += temp;
    }
    
    System.out.println ();
    
    // get the time again
    msec = Calendar.getInstance().getTimeInMillis() - msec;
    
    // and vertify the total
    checkCloseEnough (total, 1000000.0 * 1000001.0 / 2.0, -1);
    
    // and return the time
    return msec;  
  }
  
  /**
   * Make sure that a single value can be correctly stored and extracted.
   */
  private void superSimpleTester (ISparseArray <Double> myArray) {
    myArray.put (12, 3.4);
    if (myArray.get (0) != null || myArray.get (14) != null) 
      fail ("Expected a null!");
    checkCloseEnough (myArray.get (12), 3.4, 12);
  }

  /**
   * This puts some data in ascending order, then puts some data in the front
   * of the array in ascending order, and then makes sure it can all come
   * out once again.
   */
  private void moreComplexTester (ISparseArray <Double> myArray) {
    
    for (int i = 100; i < 10000; i++) {
      myArray.put (i * 10, i * 1.0);
    }
    for (int i = 0; i < 100; i++) {
      myArray.put (i * 10, i * 1.0);
    }
    
    for (int i = 0; i < 10000; i++) {
      if (myArray.get (i * 10 + 1) != null)
        fail ("Expected a null at pos " + i * 10 + 1);
    }
   
    for (int i = 100; i < 10000; i++) {
      checkCloseEnough (myArray.get (i * 10), i * 1.0, i * 10);
    }
    for (int i = 99; i >= 0; i--) {
      checkCloseEnough (myArray.get (i * 10), i * 1.0, i * 10);
    }
    
  }
  
  /**
   * This puts data in using a weird order, and then extracts it using another 
   * weird order
   */
  private void strangeOrderTester (ISparseArray <Double> myArray) {
    
    for (int i = 0; i < 500; i++) {
      myArray.put (500 + i, (500 + i) * 4.99);
      myArray.put (499 - i, (499 - i) * 5.99);
    }
  
    for (int i = 0; i < 500; i++) {
      checkCloseEnough (myArray.get (i), i * 5.99, i);
      checkCloseEnough (myArray.get (999 - i), (999 - i) * 4.99, 999 - i);
    }
  }
    
  /**
   * Adds 1000 values into the array, and then sees if they are correctly extracted
   * using an iterator.
   */
  private void iteratorTester (ISparseArray <Double> myArray) {
    
    // put a bunch of values in
    for (int i = 0; i < 1000; i++) {
      myArray.put (i * 100, i * 23.45);  
    }
    
    // then create an iterator and make sure we can extract them once again
    Iterator <IIndexedData <Double>> myIter = myArray.iterator ();
    int counter = 0;
    while (myIter.hasNext ()) {
      IIndexedData <Double> curOne = myIter.next ();
      if (curOne.getIndex () != counter * 100)
        fail ("I was using an iterator; expected index " + counter * 100 + " but got " + curOne.getIndex ());
      checkCloseEnough (curOne.getData (), counter * 23.45, counter * 100);
      counter++;
    }
  }
    
  /** 
   * Used to see if the array can correctly accept writes and rewrites of 
   * the same slots, and get the re-written values back
   */
  private void rewriteTester (ISparseArray <Double> myArray) {
    for (int i = 0; i < 100; i++) {
      myArray.put (i * 4, i * 9.34);  
    }
    for (int i = 99; i >= 0; i--) {
      myArray.put (i * 4, i * 19.34);  
    }
    for (int i = 0; i < 100; i++) {
      checkCloseEnough (myArray.get (i * 4), i * 19.34, i * 4);
    } 
  }
  
  /**
   * Checks both arrays to make sure that if there is no data, you can
   * still correctly create and use an iterator.
   */
  public void testEmptyIterator () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    Iterator <IIndexedData <Double>> myIter = myArray.iterator ();
    if (myIter.hasNext ()) {
      fail ("The linear sparse array should have had no data!");
    }
    myArray = new TreeSparseArray <Double> ();
    myIter = myArray.iterator ();
    if (myIter.hasNext ()) {
      fail ("The tree sparse array should have had no data!");
    }
  }
  
  /**
   * Checks to see if you can put a bunch of data in, and then get it out
   * again using an iterator.
   */
  public void testIteratorLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    iteratorTester (myArray);
  }
  
  public void testIteratorTree () {
    ISparseArray <Double> myArray = new TreeSparseArray <Double> ();
    iteratorTester (myArray);
  }
    
  /**
   * Tests whether one can write data into the array, then write over it with
   * new data, and get the re-written values out once again
   */
  public void testRewriteLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    rewriteTester (myArray);
  }
  
  public void testRewriteTree () {
    ISparseArray <Double> myArray = new TreeSparseArray <Double> ();
    rewriteTester (myArray);
  }
 
  /** Inserts in ascending and then in descending order, and then does a lot
    * of gets to check to make sure the correct data is in there
    */
  public void testMoreComplexLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    moreComplexTester (myArray);
  }
  
  public void testMoreComplexTree () {
    ISparseArray <Double> myArray = new TreeSparseArray <Double> ();
    moreComplexTester (myArray);
  }
 
  /**
   * Tests a single insert; tries to retreive that insert, and checks a couple
   * of slots for null.
   */
  public void testSuperSimpleInsertTree () {
    ISparseArray <Double> myArray = new TreeSparseArray <Double> ();
    superSimpleTester (myArray);
  }
    
  public void testSuperSimpleInsertLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    superSimpleTester (myArray);
  }
    
  /**
   * This does a bunch of backwards-order puts and gets into a linear sparse array
   */
  public void testBackwardsInsertLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    
    for (int i = 1000; i < 0; i--)
      myArray.put (i, i * 2.0);
    
    for (int i = 1000; i < 0; i--)
      checkCloseEnough (myArray.get (i), i * 2.0, i);
  }
  
  /**
   * These do a bunch of gets in puts in a non-linear order
   */
  public void testStrageOrderLinear () {
    ISparseArray <Double> myArray = new LinearSparseArray <Double> (2000);
    strangeOrderTester (myArray);
  }
  
  public void testStrageOrdeTree () {
    ISparseArray <Double> myArray = new TreeSparseArray <Double> ();
    strangeOrderTester (myArray);
  }
  
  /**
   * Used to check the raw speed of sequential access in the implementation
   */
  public void speedCheck (double goal) {
    
    // create two sparse arrays
    ISparseArray <Double> myArray1 = new SparseArray <Double> ();
    ISparseArray <Double> myArray2 = new LinearSparseArray <Double> (2000);
    
    // test the first one
    System.out.println ("**************** SPEED TEST ******************");
    System.out.println ("Testing the provided, tree-based sparse array.");
    long firstTime = doInsertThenScan (myArray1);
    
    // test the second one
    System.out.println ("Testing the linear sparse array.");
    long secTime = doInsertThenScan (myArray2);
    
    System.out.format ("The linear took %g%% as long as the tree-based\n", (secTime * 100.0) / firstTime);
    System.out.format ("The goal is %g%% or less.\n", goal);
    
    if ((secTime * 100.0) / firstTime > goal)
      fail ("Not fast enough to pass the speed check!");
    
    System.out.println ("**********************************************");
  }
  
  /**
   * See if the linear sparse array takes less than 2/3 of the time comapred to SparseArray
   */
  public void testSpeedMedium () {
    speedCheck (66.67);  
  }
  
  /**
   * See if the linear sparse array takes less than 1/2 of the time comapred to SparseArray
   */
  public void testSpeedFast () {
    speedCheck (50.0);  
  }
}
