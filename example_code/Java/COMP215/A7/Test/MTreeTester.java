import junit.framework.TestCase;
import java.util.ArrayList;
import java.util.Random;
import java.util.Collections;

// this is a map from a set of keys of type PointInMetricSpace to a
// set of data of type DataType
interface IMTree <Key extends IPointInMetricSpace <Key>, Data> {
 
  // insert a new key/data pair into the map
  void insert (Key keyToAdd, Data dataToAdd);
  
  // find all of the key/data pairs in the map that fall within a
  // particular distance of query point if no results are found, then
  // an empty array is returned (NOT a null!)
  ArrayList <DataWrapper <Key, Data>> find (Key query, double distance);
  
  // find the k closest key/data pairs in the map to a particular
  // query point 
  //
  // if the number of points in the map is less than k, then the
  // returned list will have less than k pairs in it
  //
  // if the number of points is zero, then an empty array is returned
  // (NOT a null!)
  ArrayList <DataWrapper <Key, Data>> findKClosest (Key query, int k);
  
  // returns the number of nodes that exist on a path from root to leaf in the tree...
  // whtever the details of the implementation are, a tree with a single leaf node should return 1, and a tree
  // with more than one lead node should return at least two (since it must have at least one internal node).
  public int depth ();
  
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


// this correponds to some geometric object in a metric space; the object is built out of
// one or more points of type PointInMetricSpace
interface IObjectInMetricSpaceABCD <PointInMetricSpace extends IPointInMetricSpace<PointInMetricSpace>> {
  
  // get the maximum distance from some point on the shape to a particular point in the metric space
  double maxDistanceTo (PointInMetricSpace checkMe);
  
  // return some arbitrary point on the interior of the geometric object
  PointInMetricSpace getPointInInterior ();
}


// this interface corresponds to a point in some metric space
interface IPointInMetricSpace <PointInMetricSpace> {
  
  // get the distance to another point
  // 
  // for this to work in an M-Tree, distances should be "metric" and
  // obey the triangle inequality
  double getDistance (PointInMetricSpace toMe);
}

// this is the basic node type in the structure
interface IMTreeNodeABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, DataType> {
  
  // insert a new key/data pair into the structure.  The return value is a new MTreeNode if there was
  // a split in response to the insertion, or a null if there was no split
  public IMTreeNodeABCD <PointInMetricSpace, DataType> insert (PointInMetricSpace keyToAdd, DataType dataToAdd);
  
  // find all of the key/data pairs in the structure that fall within a particular query sphere
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> find (SphereABCD <PointInMetricSpace> query);  
  
  // find the set of items closest to the given query point
  public void findKClosest (PointInMetricSpace query, PQueueABCD <PointInMetricSpace, DataType> myQ);
    
  // returns a sphere that totally encompasses everything in the node and its subtree
  public SphereABCD <PointInMetricSpace> getBoundingSphere ();
 
  // returns the depth
  public int depth ();
}

// this is a point in a particular metric space
class PointABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>> implements
  IObjectInMetricSpaceABCD <PointInMetricSpace> {

  // the point
  private PointInMetricSpace me;
  
  public PointInMetricSpace getPointInInterior () {
    return me; 
  }
  
  public double maxDistanceTo (PointInMetricSpace checkMe) {
    return checkMe.getDistance (me); 
  }
  
  // contruct a point
  public PointABCD (PointInMetricSpace useMe) {
    me = useMe; 
  }
}

class PQueueABCD <PointInMetricSpace, DataType> {
 
  // this is an object in the queue
  private class Wrapper {
    DataWrapper <PointInMetricSpace, DataType> myData;
    double distance;
  }
  
  // this is the actual queue
  ArrayList <Wrapper> myList;
  
  // this is the largest distance value currently in the queue
  double large;
  
  // this is the max number of objects
  int maxCount;
  
  // create a priority queue with the speced number of slots
  PQueueABCD (int maxCountIn) {
    maxCount = maxCountIn;
    myList = new ArrayList <Wrapper> ();
    large = 9e99;
  }
  
  // get the largest distance in the queue
  double getDist () {
    return large;
  }
  
  // add a new object to the queue... the insertion is ignored if the distance value
  // exceeds the largest that is in the queue and the queue is already full
  void insert (PointInMetricSpace key, DataType data, double distance) {
  
    // if we are full, remove the one with the largest distance
    if (myList.size () == maxCount && distance < large) {
      
      int badIndex = 0;
      double maxDist = 0.0;
      for (int i = 0; i < myList.size (); i++) {
        if (myList.get (i).distance > maxDist) {
          maxDist =  myList.get (i).distance;
          badIndex = i;
        }
      }
      
      myList.remove (badIndex);
    }
    
    // see if there is room
    if (myList.size () < maxCount) {
      
      // add it 
      Wrapper newOne = new Wrapper ();
      newOne.myData = new DataWrapper <PointInMetricSpace, DataType> (key, data);
      newOne.distance = distance;
      myList.add (newOne);      
    }
    
    // if we are full, reset the max distance
    if (myList.size () == maxCount) {
      large = 0.0;
      for (int i = 0; i < myList.size (); i++) {
        if (myList.get (i).distance > large) {
          large = myList.get (i).distance;
        }
      }
    }
    
  }
  
  // extracts the contents of the queue
  ArrayList <DataWrapper <PointInMetricSpace, DataType>> done () {
  
    ArrayList <DataWrapper <PointInMetricSpace, DataType>> returnVal = new ArrayList <DataWrapper <PointInMetricSpace, DataType>> ();
    for (int i = 0; i < myList.size (); i++) {
      returnVal.add (myList.get (i).myData); 
    }
    
    return returnVal;
  }
  
}

// this is a sphere in a particular metric space
class SphereABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>> implements
  IObjectInMetricSpaceABCD <PointInMetricSpace> {

  // the center of the sphere and its radius
  private PointInMetricSpace center;
  private double radius;
  
  // build a sphere out of its center and its radius
  public SphereABCD (PointInMetricSpace centerIn, double radiusIn) {
    center = centerIn;
    radius = radiusIn;
  }
    
  // increase the size of the sphere so it contains the object swallowMe
  public void swallow (IObjectInMetricSpaceABCD <PointInMetricSpace> swallowMe) {
    
    // see if we need to increase the radius to swallow this guy
    double dist = swallowMe.maxDistanceTo (center);
    if (dist > radius)
      radius = dist;
  }
  
  // check to see if a sphere intersects another sphere
  public boolean intersects (SphereABCD <PointInMetricSpace> me) {
    return (me.center.getDistance (center) <= me.radius + radius);
  }
  
  // check to see if the sphere contains a point
  public boolean contains (PointInMetricSpace checkMe) {
    return (checkMe.getDistance (center) <= radius);
  }
  
  public PointInMetricSpace getPointInInterior () {
    return center; 
  }
  
  public double maxDistanceTo (PointInMetricSpace checkMe) {
    return radius + checkMe.getDistance (center); 
  }
}



class OneDimPoint implements IPointInMetricSpace <OneDimPoint> {
 
  // this is the actual double
  Double value;
  
  // construct this out of a double value
  public OneDimPoint (double makeFromMe) {
    value = new Double (makeFromMe);
  }
  
  // get the distance to another point
  public double getDistance (OneDimPoint toMe) {
    if (value > toMe.value)
      return value - toMe.value;
    else
      return toMe.value - value;
  }
  
}

class MultiDimPoint implements IPointInMetricSpace <MultiDimPoint> {
 
  // this is the point in a multidimensional space
  IDoubleVector me;
  
  // make this out of an IDoubleVector
  public MultiDimPoint (IDoubleVector useMe) {
    me = useMe;
  }
  
  // get the Euclidean distance to another point
  public double getDistance (MultiDimPoint toMe) {
    double distance = 0.0;
    try {
      for (int i = 0; i < me.getLength (); i++) {
        double diff = me.getItem (i) - toMe.me.getItem (i);
        diff *= diff;
        distance += diff;
      }
    } catch (OutOfBoundsException e) {
        throw new IndexOutOfBoundsException ("Can't compare two MultiDimPoint objects with different dimensionality!"); 
    }
    return Math.sqrt(distance);
  }
  
}
// this is a leaf node
class LeafMTreeNodeABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, DataType> implements 
  IMTreeNodeABCD <PointInMetricSpace, DataType> {
 
  // this holds all of the data in the leaf node
  private IMetricDataListABCD <PointInMetricSpace, PointABCD <PointInMetricSpace>, DataType> myData;
  
  // contructor that takes a data list and builds a node out of it
  private LeafMTreeNodeABCD (IMetricDataListABCD <PointInMetricSpace, PointABCD <PointInMetricSpace>, DataType> myDataIn) {
    myData = myDataIn;  
  }
  
  // constructor that creates an empty node
  public LeafMTreeNodeABCD () {
    myData = new ChrisMetricDataListABCD <PointInMetricSpace, PointABCD <PointInMetricSpace>, DataType> (); 
  }
  
  // get the sphere that bounds this node
  public SphereABCD <PointInMetricSpace> getBoundingSphere () {
    return myData.getBoundingSphere ();  
  }
  
  public IMTreeNodeABCD <PointInMetricSpace, DataType> insert (PointInMetricSpace keyToAdd, DataType dataToAdd) {
    
    // just add in the new data
    myData.add (new PointABCD <PointInMetricSpace> (keyToAdd), dataToAdd);
    
    // if we exceed the max size, then split and create a new node
    if (myData.size () > getMaxSize ()) {
       IMetricDataListABCD <PointInMetricSpace, PointABCD <PointInMetricSpace>, DataType> newOne = myData.splitInTwo ();
       LeafMTreeNodeABCD <PointInMetricSpace, DataType> returnVal = new LeafMTreeNodeABCD <PointInMetricSpace, DataType> (newOne);
       return returnVal;
    }
    
    // otherwise, we return a null to indcate that a split did not occur
    return null;
  }
  
  public void findKClosest (PointInMetricSpace query, PQueueABCD <PointInMetricSpace, DataType> myQ) {
    for (int i = 0; i < myData.size (); i++) {
      SphereABCD <PointInMetricSpace> mySphere = new SphereABCD <PointInMetricSpace> (query, myQ.getDist ());
      if (mySphere.contains (myData.get (i).getKey ().getPointInInterior ())) {
        myQ.insert (myData.get (i).getKey ().getPointInInterior (), myData.get (i).getData (), 
                 query.getDistance (myData.get (i).getKey ().getPointInInterior ()));
      }
    }
  }
  
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> find (SphereABCD <PointInMetricSpace> query) {
     
    ArrayList <DataWrapper <PointInMetricSpace, DataType>> returnVal = new ArrayList <DataWrapper <PointInMetricSpace, DataType>> ();
    
    // just search thru every entry and look for a match... add every match into the return val
    for (int i = 0; i < myData.size (); i++) {
      if (query.contains (myData.get (i).getKey ().getPointInInterior ())) {
        returnVal.add (new DataWrapper <PointInMetricSpace, DataType> (myData.get (i).getKey ().getPointInInterior (), myData.get (i).getData ()));
      }
    }
    
    return returnVal;
  }
  
  public int depth () {
    return 1; 
  }

  // this is the max number of entries in the node
  private static int maxSize;
  
  // and a getter and setter
  public static void setMaxSize (int toMe) {
    maxSize = toMe;  
  }
  public static int getMaxSize () {
    return maxSize;
  }
}

// this silly little class is just a wrapper for a key, data pair
class DataWrapper <Key, Data> {
  
  Key key;
  Data data;
  
  public DataWrapper (Key keyIn, Data dataIn) {
    key = keyIn;
    data = dataIn;
  }
  
  public Key getKey () {
    return key;
  }
  
  public Data getData () {
    return data;
  }
}


// this is an internal node
class InternalMTreeNodeABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, DataType>
  implements IMTreeNodeABCD <PointInMetricSpace, DataType> {
  
  // this holds all of the data in the leaf node
  private IMetricDataListABCD <PointInMetricSpace, SphereABCD <PointInMetricSpace>, IMTreeNodeABCD <PointInMetricSpace, DataType>> myData;
  
  // get the sphere that bounds this node
  public SphereABCD <PointInMetricSpace> getBoundingSphere () {
    return myData.getBoundingSphere ();  
  }
  
  // this constructs a new internal node that holds precisely the two nodes that are passed in
  public InternalMTreeNodeABCD (IMTreeNodeABCD <PointInMetricSpace, DataType> refOne,
                            IMTreeNodeABCD <PointInMetricSpace, DataType> refTwo) {
    
    // create a necw list and add the two guys in
    myData = new ChrisMetricDataListABCD <PointInMetricSpace, SphereABCD <PointInMetricSpace>, IMTreeNodeABCD <PointInMetricSpace, DataType>> ();
    myData.add (refOne.getBoundingSphere (), refOne);
    myData.add (refTwo.getBoundingSphere (), refTwo);
  }
  
  // this constructs a new node that holds the list of data that we were passed in
  public InternalMTreeNodeABCD (IMetricDataListABCD <PointInMetricSpace, SphereABCD <PointInMetricSpace>, IMTreeNodeABCD <PointInMetricSpace, DataType>> useMe) {
     myData = useMe; 
  }
  
  // from the interface
  public IMTreeNodeABCD <PointInMetricSpace, DataType> insert (PointInMetricSpace keyToAdd, DataType dataToAdd) {
    
    // find the best child; this is the one we are closest to
    double bestDist = 9e99;
    int bestIndex = 0;
    for (int i = 0; i < myData.size (); i++) {
    
      double newDist = myData.get (i).getKey ().getPointInInterior ().getDistance (keyToAdd);
      if (newDist < bestDist) {
        bestDist = newDist;
        bestIndex = i;
      }
    }
    
    // do the recursive insert
    IMTreeNodeABCD <PointInMetricSpace, DataType> newRef = myData.get (bestIndex).getData ().insert (keyToAdd, dataToAdd);
    
    // if we got something back, then there was a split, so add the new node into our list
    if (newRef != null) {
      myData.add (newRef.getBoundingSphere (), newRef);
    }
    
    // if we exceed the max size, then split and create a new node
    if (myData.size () > getMaxSize ()) {
       IMetricDataListABCD <PointInMetricSpace, SphereABCD <PointInMetricSpace>, IMTreeNodeABCD <PointInMetricSpace, DataType>> newOne = myData.splitInTwo ();
       InternalMTreeNodeABCD <PointInMetricSpace, DataType> returnVal = new InternalMTreeNodeABCD <PointInMetricSpace, DataType> (newOne);
       return returnVal;
    }
    
    // otherwise, we return a null to indcate that a split did not occur
    return null;
  }
  
  public void findKClosest (PointInMetricSpace query, PQueueABCD <PointInMetricSpace, DataType> myQ) {
    for (int i = 0; i < myData.size (); i++) {
      SphereABCD <PointInMetricSpace> mySphere = new SphereABCD <PointInMetricSpace> (query, myQ.getDist ());
      if (mySphere.intersects (myData.get (i).getKey ())) {
        myData.get (i).getData ().findKClosest (query, myQ);
      }
    }
  }
  
  // from the interface
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> find (SphereABCD <PointInMetricSpace> query) {
     
    ArrayList <DataWrapper <PointInMetricSpace, DataType>> returnVal = new ArrayList <DataWrapper <PointInMetricSpace, DataType>> ();
    
    // just search thru every entry and look for a match... add every match into the return val
    for (int i = 0; i < myData.size (); i++) {
      if (query.intersects (myData.get (i).getKey ())) {
        returnVal.addAll (myData.get (i).getData ().find (query));
      }
    }
    
    return returnVal;
  }
  
  public int depth () {
    return myData.get (0).getData ().depth () + 1; 
  }
  
  // this is the max number of entries in the node
  private static int maxSize;
  
  // and a getter and setter
  public static void setMaxSize (int toMe) {
    maxSize = toMe;  
  }
  public static int getMaxSize () {
    return maxSize;
  }
}

abstract class AMetricDataListABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, 
  KeyType extends IObjectInMetricSpaceABCD <PointInMetricSpace>, DataType> implements IMetricDataListABCD <PointInMetricSpace,KeyType, DataType> {

  // this is the data that is actually stored in the list
  private ArrayList <DataWrapper <KeyType, DataType>> myData;
  
  // this is the sphere that bounds everything in the list
  private SphereABCD <PointInMetricSpace> myBoundingSphere;
 
  public SphereABCD <PointInMetricSpace> getBoundingSphere () {
    return myBoundingSphere;  
  }
  
  public int size () {
    if (myData != null)
      return myData.size ();  
    else
      return 0;
  }
  
  public DataWrapper <KeyType, DataType> get (int pos) {
    return myData.get (pos);
  }
 
  public void replaceKey (int pos, KeyType keyToAdd) {
    DataWrapper <KeyType, DataType> temp = myData.remove (pos);
    DataWrapper <KeyType, DataType> newOne = new DataWrapper <KeyType, DataType> (keyToAdd, temp.getData ());
    myData.add (pos, newOne);
  }
  
  public void add (KeyType keyToAdd, DataType dataToAdd) {
    
    // if this is the first add, then set everyone up
    if (myData == null) {
      PointInMetricSpace myCentroid = keyToAdd.getPointInInterior ();
      myBoundingSphere = new SphereABCD <PointInMetricSpace> (myCentroid, keyToAdd.maxDistanceTo (myCentroid));
      myData = new ArrayList <DataWrapper <KeyType, DataType>> ();
    
    // otherwise, extend the sphere if needed
    } else {
      myBoundingSphere.swallow (keyToAdd);
    }
    
    // add the data
    myData.add (new DataWrapper <KeyType, DataType> (keyToAdd, dataToAdd));
  }
  
  // we want to potentially allow many implementations of the splitting lalgorithm
  abstract public IMetricDataListABCD <PointInMetricSpace, KeyType, DataType> splitInTwo ();
  
  // this is here so the actual splitting algorithn can access the list and change the sphere
  protected ArrayList <DataWrapper <KeyType, DataType>> getList () {
    return myData;
  }
  
  // this is here so the splitting algorithm can modify the list and the sphere
  protected void replaceGuts (AMetricDataListABCD <PointInMetricSpace, KeyType, DataType> withMe) {
    myData = withMe.myData;
    myBoundingSphere = withMe.myBoundingSphere;
  }
    
}

// this is a particular implementation of the metric data list that uses a simple quadratic clustering
// algorithm to perform a list split.  It picks two "seeds" (the most distant objects) and then repeatedly
// adds to the two new lists by choosing the objects closest to the seeds
class ChrisMetricDataListABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, 
  KeyType extends IObjectInMetricSpaceABCD <PointInMetricSpace>, DataType> extends AMetricDataListABCD <PointInMetricSpace, KeyType, DataType> {

  public IMetricDataListABCD <PointInMetricSpace, KeyType, DataType> splitInTwo () {
    
    // first, we need to find the two most distant points in the list
    ArrayList <DataWrapper <KeyType, DataType>> myData = getList ();
    int bestI = 0, bestJ = 0;
    double maxDist = -1.0;
    for (int i = 0; i < myData.size (); i++) {
      for (int j = i + 1; j < myData.size (); j++) {
        
        // get the two keys
        DataWrapper <KeyType, DataType> pointOne = myData.get (i);
        DataWrapper <KeyType, DataType> pointTwo = myData.get (j);
        
        // find the difference between them
        double curDist = pointOne.getKey ().getPointInInterior ().getDistance (pointTwo.getKey ().getPointInInterior ());

        // see if it is the biggest
        if (curDist > maxDist) {
           maxDist = curDist;
           bestI = i;
           bestJ = j;
        }
      }
    }
    
    // now, we have the best two points; those are seeds for the clustering
    DataWrapper <KeyType, DataType> pointOne = myData.remove (bestI);
    DataWrapper <KeyType, DataType> pointTwo = myData.remove (bestJ - 1);
    
    // these are the two new lists
    AMetricDataListABCD <PointInMetricSpace, KeyType, DataType> listOne = new ChrisMetricDataListABCD <PointInMetricSpace, KeyType, DataType> ();
    AMetricDataListABCD <PointInMetricSpace, KeyType, DataType> listTwo = new ChrisMetricDataListABCD <PointInMetricSpace, KeyType, DataType> ();
    
    // put the two seeds in
    listOne.add (pointOne.getKey (), pointOne.getData ());
    listTwo.add (pointTwo.getKey (), pointTwo.getData ());
    
    // and add everyone else in
    while (myData.size () != 0) {
      
      // find the one closest to the first seed
      int bestIndex = 0;
      double bestDist = 9e99;
      
      // loop thru all of the candidate data objects
      for (int i = 0; i < myData.size (); i++) {
        if (myData.get (i).getKey ().getPointInInterior ().getDistance (pointOne.getKey ().getPointInInterior ()) < bestDist) {
          bestDist = myData.get (i).getKey ().getPointInInterior ().getDistance (pointOne.getKey ().getPointInInterior ());
          bestIndex = i;
        }
      }
      
      // and add the best in
      listOne.add (myData.get (bestIndex).getKey (), myData.get (bestIndex).getData ());
      myData.remove (bestIndex);
            
      // break if no more data
      if (myData.size () == 0)
        break;
      
      // loop thru all of the candidate data objects
      bestDist = 9e99;
      for (int i = 0; i < myData.size (); i++) {
        if (myData.get (i).getKey ().getPointInInterior ().getDistance (pointTwo.getKey ().getPointInInterior ()) < bestDist) {
          bestDist = myData.get (i).getKey ().getPointInInterior ().getDistance (pointTwo.getKey ().getPointInInterior ());
          bestIndex = i;
        }
      }
      
      // and add the best in
      listTwo.add (myData.get (bestIndex).getKey (), myData.get (bestIndex).getData ());
      myData.remove (bestIndex);
    }
    
    // now we replace our own guts with the first list
    replaceGuts (listOne);
    
    // and return the other list
    return listTwo;
  }
}

interface IMetricDataListABCD <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, 
  KeyType extends IObjectInMetricSpaceABCD <PointInMetricSpace>, DataType> {
  
  // add a new pair to the list
  public void add (KeyType keyToAdd, DataType dataToAdd);
  
  // replace a key at the indicated position
  public void replaceKey (int pos, KeyType keyToAdd);
  
  // get the key, data pair that resides at a particular position
  public DataWrapper <KeyType, DataType> get (int pos);
  
  // return the number of items in the list
  public int size ();
  
  // split the list in two, using some clutering algorithm
  public IMetricDataListABCD <PointInMetricSpace, KeyType, DataType> splitInTwo ();
  
  // get a sphere that totally bounds all of the objects in the list
  public SphereABCD <PointInMetricSpace> getBoundingSphere ();
}

// this implements a map from a set of keys of type PointInMetricSpace to a set of data of type DataType
class MTree <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, DataType> implements IMTree <PointInMetricSpace, DataType> {
  
  // this is the actual tree
  private IMTreeNodeABCD <PointInMetricSpace, DataType> root;
  
  // constructor... the two params set the number of entries in leaf and internal nodes
  public MTree (int intNodeSize, int leafNodeSize) {
    InternalMTreeNodeABCD.setMaxSize (intNodeSize);
    LeafMTreeNodeABCD.setMaxSize (leafNodeSize);
    root = new LeafMTreeNodeABCD <PointInMetricSpace, DataType> ();
  }
  
  // insert a new key/data pair into the map
  public void insert (PointInMetricSpace keyToAdd, DataType dataToAdd) {
    
    // insert the new data point
    IMTreeNodeABCD <PointInMetricSpace, DataType> res = root.insert (keyToAdd, dataToAdd);
    
    // if we got back a root split, then construct a new root
    if (res != null) {
      root = new InternalMTreeNodeABCD <PointInMetricSpace, DataType> (root, res);
    }
  }
  
  // find all of the key/data pairs in the map that fall within a particular distance of query point
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> find (PointInMetricSpace query, double distance) {
    return root.find (new SphereABCD <PointInMetricSpace> (query, distance)); 
  }
  
  // find the k closest key/data pairs in the map to a particular query point
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> findKClosest (PointInMetricSpace query, int k) {
    PQueueABCD <PointInMetricSpace, DataType> myQ = new PQueueABCD <PointInMetricSpace, DataType> (k);
    root.findKClosest (query, myQ);
    return myQ.done ();
  }
  
  // get the depth
  public int depth () {
    return root.depth (); 
  }
}

// this implements a map from a set of keys of type PointInMetricSpace to a set of data of type DataType
class SCMTree <PointInMetricSpace extends IPointInMetricSpace <PointInMetricSpace>, DataType> implements IMTree <PointInMetricSpace, DataType> {
 
  // this is the actual tree
  private IMTreeNodeABCD <PointInMetricSpace, DataType> root;
  
  // constructor... the two params set the number of entries in leaf and internal nodes
  public SCMTree (int intNodeSize, int leafNodeSize) {
    InternalMTreeNodeABCD.setMaxSize (intNodeSize);
    LeafMTreeNodeABCD.setMaxSize (leafNodeSize);
    root = new LeafMTreeNodeABCD <PointInMetricSpace, DataType> ();
  }
  
  // insert a new key/data pair into the map
  public void insert (PointInMetricSpace keyToAdd, DataType dataToAdd) {
    
    // insert the new data point
    IMTreeNodeABCD <PointInMetricSpace, DataType> res = root.insert (keyToAdd, dataToAdd);
    
    // if we got back a root split, then construct a new root
    if (res != null) {
      root = new InternalMTreeNodeABCD <PointInMetricSpace, DataType> (root, res);
    }
  }
  
  // find all of the key/data pairs in the map that fall within a particular distance of query point
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> find (PointInMetricSpace query, double distance) {
    return root.find (new SphereABCD <PointInMetricSpace> (query, distance)); 
  }
  
  // find the k closest key/data pairs in the map to a particular query point
  public ArrayList <DataWrapper <PointInMetricSpace, DataType>> findKClosest (PointInMetricSpace query, int k) {
    PQueueABCD <PointInMetricSpace, DataType> myQ = new PQueueABCD <PointInMetricSpace, DataType> (k);
    root.findKClosest (query, myQ);
    return myQ.done ();
  }
  
  // get the depth
  public int depth () {
    return root.depth (); 
  }
}


/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */

public class MTreeTester extends TestCase {
  
  // compare two lists of strings to see if the contents are the same
  private boolean compareStringSets (ArrayList <String> setOne, ArrayList <String> setTwo) {
        
    // sort the sets and make sure they are the same
    boolean returnVal = true;
    if (setOne.size () == setTwo.size ()) {
      Collections.sort (setOne);
      Collections.sort (setTwo);
      for (int i = 0; i < setOne.size (); i++) {
        if (!setOne.get (i).equals (setTwo.get (i))) {
          returnVal = false;
          break; 
        }
      }
    } else {
      returnVal = false;
    }
    
    // print out the result
    System.out.println ("**** Expected:");
    for (int i = 0; i < setOne.size (); i++) {
      System.out.format (setOne.get (i) + " ");
    }
    System.out.println ("\n**** Got:");
    for (int i = 0; i < setTwo.size (); i++) {
      System.out.format (setTwo.get (i) + " ");
    }
    System.out.println ("");
    
    return returnVal;
  }
  
  
  /**
   * THESE TWO HELPER METHODS TEST SIMPLE ONE DIMENSIONAL DATA
   */
  
  // puts the specified number of random points into a tree of the given node size
  private void testOneDimRangeQuery (boolean orderThemOrNot, int minDepth,
                                               int internalNodeSize, int leafNodeSize, int numPoints, double expectedQuerySize) {
    
    // create two trees
    Random rn = new Random (133);
    IMTree <OneDimPoint, String> myTree = new SCMTree <OneDimPoint, String> (internalNodeSize, leafNodeSize);
    IMTree <OneDimPoint, String> hisTree = new MTree <OneDimPoint, String> (internalNodeSize, leafNodeSize);
    
    // add a bunch of points
    if (orderThemOrNot) {
      for (int i = 0; i < numPoints; i++) {
        double nextOne = i / (numPoints * 1.0);
        myTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
        hisTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
      }
    } else {
      for (int i = 0; i < numPoints; i++) {
        double nextOne = rn.nextDouble ();
        myTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
        hisTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
      }      
    }
    
    assertTrue ("Why was the tree not of depth at least " + minDepth, hisTree.depth () >= minDepth);
    
    // run a range query
    OneDimPoint query = new OneDimPoint (numPoints * 0.5);
    ArrayList <DataWrapper <OneDimPoint, String>> result1 = myTree.find (query, expectedQuerySize / 2);
    ArrayList <DataWrapper <OneDimPoint, String>> result2 = hisTree.find (query, expectedQuerySize / 2);
    
    // check the results
    ArrayList <String> setOne = new ArrayList <String> (), setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
    
    // run a second range query
    query = new OneDimPoint (numPoints);
    result1 = myTree.find (query, expectedQuerySize);
    result2 = hisTree.find (query, expectedQuerySize);
    
    // check the results
    setOne = new ArrayList <String> ();
    setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
  }
  
  // like the last one, but runs a top k
  private void testOneDimTopKQuery (boolean orderThemOrNot, int minDepth,
                                              int internalNodeSize, int leafNodeSize, int numPoints, int querySize) {
    
    // create two trees
    Random rn = new Random (167);
    IMTree <OneDimPoint, String> myTree = new SCMTree <OneDimPoint, String> (internalNodeSize, leafNodeSize);
    IMTree <OneDimPoint, String> hisTree = new MTree <OneDimPoint, String> (internalNodeSize, leafNodeSize);
    
    // add a bunch of points
    if (orderThemOrNot) {
      for (int i = 0; i < numPoints; i++) {
        double nextOne = i / (numPoints * 1.0);
        myTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
        hisTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
      }
    } else {
      for (int i = 0; i < numPoints; i++) {
        double nextOne = rn.nextDouble ();
        myTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
        hisTree.insert (new OneDimPoint (nextOne * numPoints), (new Double (nextOne * numPoints)).toString ());
      }      
    }
    
    assertTrue ("Why was the tree not of depth at least " + minDepth, hisTree.depth () >= minDepth);
        
    // run a range query
    OneDimPoint query = new OneDimPoint (numPoints * 0.5);
    ArrayList <DataWrapper <OneDimPoint, String>> result1 = myTree.findKClosest (query, querySize);
    ArrayList <DataWrapper <OneDimPoint, String>> result2 = hisTree.findKClosest (query, querySize);
    
    // check the results
    ArrayList <String> setOne = new ArrayList <String> (), setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
    
    // run a second range query
    query = new OneDimPoint (0);
    result1 = myTree.findKClosest (query, querySize);
    result2 = hisTree.findKClosest (query, querySize);
    
    // check the results
    setOne = new ArrayList <String> ();
    setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
  }
  
  /**
   * THESE TWO HELPER METHODS TEST MULTI-DIMENSIONAL DATA
   */
    
  // puts the specified number of random points into a tree of the given node size
  private void testMultiDimRangeQuery (boolean orderThemOrNot, int minDepth, int numDims,
                                               int internalNodeSize, int leafNodeSize, int numPoints, double expectedQuerySize) {
    
    // create two trees
    Random rn = new Random (133);
    IMTree <MultiDimPoint, String> myTree = new SCMTree <MultiDimPoint, String> (internalNodeSize, leafNodeSize);
    IMTree <MultiDimPoint, String> hisTree = new MTree <MultiDimPoint, String> (internalNodeSize, leafNodeSize);
    
    // these are the queries
    double dist1 = 0;
    double dist2 = 0;
    MultiDimPoint query1;
    MultiDimPoint query2;
    
    // add a bunch of points
    if (orderThemOrNot) {
      
      for (int i = 0; i < numPoints; i++) {
        SparseDoubleVector temp1 = new SparseDoubleVector (numDims, i);
        SparseDoubleVector temp2 = new SparseDoubleVector (numDims, i);
        myTree.insert (new MultiDimPoint (temp1), temp1.toString ());
        hisTree.insert (new MultiDimPoint (temp2), temp2.toString ());
      }
      
      // in this case, the queries are simple
      query1 = new MultiDimPoint (new SparseDoubleVector (numDims, numPoints / 2));
      query2 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.0));
      dist1 = Math.sqrt (numDims) * expectedQuerySize / 2.0;
      dist2 = Math.sqrt (numDims) * expectedQuerySize;
      
    } else {
      
      // create a bunch of random data
      for (int i = 0; i < numPoints; i++) {
        DenseDoubleVector temp1 = new DenseDoubleVector (numDims, 0.0);
        DenseDoubleVector temp2 = new DenseDoubleVector (numDims, 0.0);
        for (int j = 0; j < numDims; j++) {
          double curVal = rn.nextDouble ();
          try {
            temp1.setItem (j, curVal);
            temp2.setItem (j, curVal);
          } catch (OutOfBoundsException e) {
            System.out.println ("Error in test case?  Why am I out of bounds?"); 
          }
        }
        myTree.insert (new MultiDimPoint (temp1), temp1.toString ());
        hisTree.insert (new MultiDimPoint (temp2), temp2.toString ());
      }
      
      // in this case, get the spheres first
      query1 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.5));
      query2 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.0));
      
      // do a top k query
      ArrayList <DataWrapper <MultiDimPoint, String>> result1 = myTree.findKClosest (query1, (int) expectedQuerySize);
      ArrayList <DataWrapper <MultiDimPoint, String>> result2 = myTree.findKClosest (query2, (int) expectedQuerySize);
      
      // find the distance to the furthest point in both cases
      for (int i = 0; i < (int) expectedQuerySize; i++) {
        double newDist = result1.get (i).getKey ().getDistance (query1);
        if (newDist > dist1)
          dist1 = newDist;
        newDist = result2.get (i).getKey ().getDistance (query2);
        if (newDist > dist2)
          dist2 = newDist;
      }
    }
    
    assertTrue ("Why was the tree not of depth at least " + minDepth, hisTree.depth () >= minDepth);
    
    // run a range query
    ArrayList <DataWrapper <MultiDimPoint, String>> result1 = myTree.find (query1, dist1);
    ArrayList <DataWrapper <MultiDimPoint, String>> result2 = hisTree.find (query1, dist1);
    
    // check the results
    ArrayList <String> setOne = new ArrayList <String> (), setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
    
    // run a second range query
    result1 = myTree.find (query2, dist2);
    result2 = hisTree.find (query2, dist2);
    
    // check the results
    setOne = new ArrayList <String> ();
    setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
  }
  
  // puts the specified number of random points into a tree of the given node size
  private void testMultiDimTopKQuery (boolean orderThemOrNot, int minDepth, int numDims,
                                               int internalNodeSize, int leafNodeSize, int numPoints, int querySize) {
    
    // create two trees
    Random rn = new Random (133);
    IMTree <MultiDimPoint, String> myTree = new SCMTree <MultiDimPoint, String> (internalNodeSize, leafNodeSize);
    IMTree <MultiDimPoint, String> hisTree = new MTree <MultiDimPoint, String> (internalNodeSize, leafNodeSize);
    
    // these are the queries
    MultiDimPoint query1;
    MultiDimPoint query2;
    
    // add a bunch of points
    if (orderThemOrNot) {
      
      for (int i = 0; i < numPoints; i++) {
        SparseDoubleVector temp1 = new SparseDoubleVector (numDims, i);
        SparseDoubleVector temp2 = new SparseDoubleVector (numDims, i);
        myTree.insert (new MultiDimPoint (temp1), temp1.toString ());
        hisTree.insert (new MultiDimPoint (temp2), temp2.toString ());
      }
      
      // in this case, the queries are simple
      query1 = new MultiDimPoint (new SparseDoubleVector (numDims, numPoints / 2));
      query2 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.0));
      
    } else {
      
      // create a bunch of random data
      for (int i = 0; i < numPoints; i++) {
        DenseDoubleVector temp1 = new DenseDoubleVector (numDims, 0.0);
        DenseDoubleVector temp2 = new DenseDoubleVector (numDims, 0.0);
        for (int j = 0; j < numDims; j++) {
          double curVal = rn.nextDouble ();
          try {
            temp1.setItem (j, curVal);
            temp2.setItem (j, curVal);
          } catch (OutOfBoundsException e) {
            System.out.println ("Error in test case?  Why am I out of bounds?"); 
          }
        }
        myTree.insert (new MultiDimPoint (temp1), temp1.toString ());
        hisTree.insert (new MultiDimPoint (temp2), temp2.toString ());
      }
      
      // in this case, get the spheres first
      query1 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.5));
      query2 = new MultiDimPoint (new SparseDoubleVector (numDims, 0.0));
    }
    
    assertTrue ("Why was the tree not of depth at least " + minDepth, hisTree.depth () >= minDepth);
    
    // run a top k query
    ArrayList <DataWrapper <MultiDimPoint, String>> result1 = myTree.findKClosest (query1, querySize);
    ArrayList <DataWrapper <MultiDimPoint, String>> result2 = hisTree.findKClosest (query1, querySize);
    
    // check the results
    ArrayList <String> setOne = new ArrayList <String> (), setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
    
    // run a second range query
    result1 = myTree.findKClosest (query2, querySize);
    result2 = hisTree.findKClosest (query2, querySize);
    
    // check the results
    setOne = new ArrayList <String> ();
    setTwo = new ArrayList <String> ();
    for (int i = 0; i < result1.size (); i++) {
      setOne.add (result1.get (i).getData ());  
    }
    for (int i = 0; i < result2.size (); i++) {
      setTwo.add (result2.get (i).getData ());  
    }
    assertTrue (compareStringSets (setOne, setTwo));
  }
  
  
  /**
   * "TIER 1" TESTS
   *  NONE OF THESE REQUIRE ANY SPLITS
   */
  public void testEasy1() {
    testOneDimRangeQuery (true, 1, 10, 10, 5, 2.1);
  }
  
  public void testEasy2() {
    testOneDimRangeQuery (false, 1, 10, 10, 5, 2.1);
  }
  
  public void testEasy3() {
    testOneDimRangeQuery (true, 1, 10000, 10000, 5000, 10);
  }
  
  public void testEasy4() {
    testOneDimRangeQuery (false, 1, 10000, 10000, 5000, 10);
  }
  
  public void testEasy5() {
    testMultiDimRangeQuery (true, 1, 5, 10, 10, 5, 2.1);
  }
  
  public void testEasy6() {
    testMultiDimRangeQuery (false, 1, 10, 10, 10, 5, 2.1);
  }
  
  public void testEasy7() {
    testMultiDimRangeQuery (true, 1, 5, 10000, 10000, 5000, 10);
  }
  
  public void testEasy8() {
    testMultiDimRangeQuery (false, 1, 15, 10000, 10000, 1000, 4);
  }
  
  /**
   * "TIER 2" TESTS
   *  NONE OF THESE REQUIRE A SPLIT OF AN INTERNAL NODE
   */
  
  public void testMod1() {
    testOneDimRangeQuery (true, 2, 10, 10, 50, 5.1);
  }
  
  public void testMod2() {
    testOneDimRangeQuery (false, 2, 10, 10, 50, 5.1);
  }
  
  public void testMod3() {
    testOneDimRangeQuery (true, 2, 20, 100, 1000, 10);
  }
  
  public void testMod4() {
    testOneDimRangeQuery (false, 2, 20, 100, 1000, 10);
  }
  
  public void testMod5() {
    testMultiDimRangeQuery (true, 2, 5, 1000, 200, 10000, 2.1);
  }
  
  public void testMod6() {
    testMultiDimRangeQuery (false, 2, 10, 1000, 200, 10000, 2.1);
  }
  
  public void testMod7() {
    testMultiDimRangeQuery (true, 2, 5, 10000, 100, 1000, 10);
  }
  
  public void testMod8() {
    testMultiDimRangeQuery (false, 2, 15, 10000, 100, 1000, 4);
  }
  
  /**
   * "TIER 3" TESTS
   *  THESE REQUIRE A SPLIT OF AN INTERNAL NODE
   */
  
  public void testHard1() {
    testOneDimRangeQuery (true, 3, 10, 10, 500, 5.1);
  }
  
  public void testHard2() {
    testOneDimRangeQuery (false, 3, 10, 10, 500, 5.1);
  }
  
  public void testHard3() {
    testOneDimRangeQuery (true, 3, 4, 4, 10000, 10);
  }
  
  public void testHard4() {
    testOneDimRangeQuery (false, 3, 4, 4, 10000, 10);
  }
  
  public void testHard5() {
    testMultiDimRangeQuery (true, 3, 5, 5, 5, 100000, 2.1);
  }
  
  public void testHard6() {
    testMultiDimRangeQuery (false, 3, 5, 5, 5, 100000, 2.1);
  }
  
  public void testHard7() {
    testMultiDimRangeQuery (true, 3, 15, 4, 4, 10000, 10);
  }
  
  public void testHard8() {
    testMultiDimRangeQuery (false, 3, 15, 4, 4, 10000, 4);
  }
  
  /**
   * "TIER 4" TESTS
   *  THESE REQUIRE A SPLIT OF AN INTERNAL NODE AND THEY TEST THE TOPK FUNCTIONALITY
   */
  
  public void testTopK1() {
    testOneDimTopKQuery (true, 3, 10, 10, 500, 5);
  }
  
  public void testTopK2() {
    testOneDimTopKQuery (false, 3, 10, 10, 500, 5);
  }
  
  public void testTopK3() {
    testOneDimTopKQuery (true, 3, 4, 4, 10000, 10);
  }
  
  public void testTopK4() {
    testOneDimTopKQuery (false, 3, 4, 4, 10000, 10);
  }
  
  public void testTopK5() {
    testMultiDimTopKQuery (true, 3, 5, 5, 5, 100000, 2);
  }
  
  public void testTopK6() {
    testMultiDimTopKQuery (false, 3, 5, 5, 5, 100000, 2);
  }
  
  public void testTopK7() {
    testMultiDimTopKQuery (true, 3, 15, 4, 4, 10000, 10);
  }
  
  public void testTopK8() {
    testMultiDimTopKQuery (false, 3, 15, 4, 4, 10000, 4);
  }
}