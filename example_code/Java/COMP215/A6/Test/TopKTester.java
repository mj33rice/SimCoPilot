import junit.framework.TestCase;
import java.util.Random;
import java.util.Collections;
import java.util.ArrayList;

// This simple interface allows us to get the top K objects out of a large set
// (that is, the K objects that are inserted having the **lowest** score vales)
// The idea is that you would create an  ITopKMachine using a specific K.  Then,
// you insert the vlaues one-by-one into the machine using the "insert" method.
// Whenever you want to obtain the current top K, you call "getTopK", which puts
// the top K into the array list.  In addition, the machine can be queried to
// see what is the current worst score that will still put a value into the top K.
interface ITopKMachine <T> {

  // insert a new value into the machine.  If its score is greater than the current
  // cutoff, it is ignored.  If its score is smaller than the current cutoff, the
  // insertion will evict the value with the worst score.
  void insert (double score, T value);
  
  // get the current top K in an array list
  ArrayList <T> getTopK ();
  
  // let the caller know the current cutoff value that will get one into the top K list
  double getCurrentCutoff ();
  
}

abstract class AVLNode <T> {
  abstract AVLNode <T> insert (double myKey, T myVal);
  abstract AVLNode <T> removeBig ();
  abstract AVLNode <T> getLeft ();
  abstract AVLNode <T> getRight ();
  abstract Data <T> getVal ();
  abstract int getHeight ();
  abstract void setHeight ();
  abstract void raiseRight ();
  abstract void raiseLeft ();
  abstract void makeLeftDeep ();
  abstract void makeRightDeep ();
  abstract void checkBalanced ();
  abstract boolean isBalanced ();
  abstract String print ();
  abstract void toList (ArrayList <T> myList);
  abstract double getBig ();
}


class EmptyNode <T> extends AVLNode <T> {
  
  void toList (ArrayList <T> myList) {}

  double getBig () {
    return Double.NEGATIVE_INFINITY;  
  }
  
  String print () {
    return "()";  
  }
  
  void checkBalanced () { 
  }
  
  AVLNode <T> insert (double myKey, T myVal) {
    return new Occupied <T> (new Data <T> (myKey, myVal), new EmptyNode <T> (), new EmptyNode <T> ());  
  }
  
  AVLNode <T> removeBig () {
    return null;  
  }
  
  AVLNode <T> getLeft () {
    return null;  
  }
  
  AVLNode <T> getRight () {
    return null;  
  }
  
  Data <T> getVal () {
    return null;  
  }
  
  int getHeight () {
    return 0;  
  }
  
  boolean isBalanced () {
    return true;
  }
  
  void setHeight () {
    throw new RuntimeException ("set height on empty node");
  }
  
  
  void raiseRight ( ){
    throw new RuntimeException ("raise right on empty node");
  } 
  
  void raiseLeft () {
    throw new RuntimeException ("raise left on empty node");
  }
  
  void makeLeftDeep () {
    throw new RuntimeException ("make left deep on empty node");
  }
  
  void makeRightDeep () {
    throw new RuntimeException ("make right deep on empty node");
  }
  
}

class Data <T> {

  private double score;
  private T val;
  
  double getScore () {
    return score;
  }
  
  T getVal () {
    return val;
  }
  
  Data (double scoreIn, T valIn) {
    score = scoreIn;
    val = valIn;
  }
}


class Occupied <T> extends AVLNode <T> {
 
  private AVLNode <T> left;
  private AVLNode <T> right;
  private Data <T> val;
  private int height;

  double getBig () {
    double big = right.getBig ();
    if (val.getScore () > big)
      return val.getScore ();
    else
      return big;
  }
  
  void toList (ArrayList <T> myList) {
    left.toList (myList);
    myList.add (val.getVal ());
    right.toList (myList);
  }
  
  String print () {
    return "(" + left.print () + val.getVal () + right.print () + ")";  
  }
  
  void checkBalanced () {
    if (!isBalanced ())
      throw new RuntimeException (left.getHeight () + " " + right.getHeight ());
    left.checkBalanced ();
    right.checkBalanced ();
  }
    
  boolean isBalanced () {
    return (!(left.getHeight () - right.getHeight () >= 2 ||
        left.getHeight () - right.getHeight () <= -2));
  }
  
  Data <T> getVal () {
    return val;  
  }
  
  AVLNode <T> removeBig () {
    AVLNode <T> newRight = right.removeBig ();
    if (newRight == null) {
      return left; 
    } else {
      right = newRight;
      setHeight ();
      
      // if we are not balanced, the RHS shrank
      if (!isBalanced ()) {
        
        // make sure the LHS is left deep
        left.makeLeftDeep ();
        
        // and now reconstruct the tree
        AVLNode <T> oldLeft = left;
        AVLNode <T> oldRight = right;
        left = oldLeft.getLeft ();
        right = new Occupied <T> (val, oldLeft.getRight (), right);
        val = oldLeft.getVal ();
        setHeight ();
      }
      
      return this;
    }
  }
  
  AVLNode <T> getLeft () {
    return left;
  }
  
  AVLNode <T> getRight () {
    return right;
  }
  
  void setHeight () {
  
    int one = left.getHeight ();
    int other = right.getHeight ();
    if (one > other)
      height = one;
    else
      height = other;
    height++;
  }
  
  int getHeight () {
    return height;  
  }
  
  void raiseLeft () {
  
    // make sure we are left deep first
    left.makeLeftDeep ();
    
    // and then balance the thing
    AVLNode <T> oldLeft = left;
    AVLNode <T> oldRight = right;
    left = oldLeft.getLeft ();
    right = new Occupied <T> (val, oldLeft.getRight (), right);
    val = oldLeft.getVal ();
    setHeight ();
  }
  
  void raiseRight () {
    
    // make sure we are right deep first
    right.makeRightDeep ();
    
    // and then balance the thing
    AVLNode <T> oldLeft = left;
    AVLNode <T> oldRight = right;
    left = new Occupied <T> (val, oldLeft, oldRight.getLeft ());
    right = oldRight.getRight ();
    val = oldRight.getVal ();
    setHeight ();
  }
  
  void makeLeftDeep () {
    
    if (left.getHeight () >= right.getHeight ())
      return;
    
    AVLNode <T> oldLeft = left;
    AVLNode <T> oldRight = right;
    left = new Occupied <T> (val, oldLeft, oldRight.getLeft ());
    right = oldRight.getRight ();
    val = oldRight.getVal ();
    setHeight ();
  }
  
  void makeRightDeep () {
    
    if (right.getHeight () >= left.getHeight ())
      return;
    
    AVLNode <T> oldLeft = left;
    AVLNode <T> oldRight = right;
    left = oldLeft.getLeft ();
    right = new Occupied <T> (val, oldLeft.getRight (), oldRight);
    val = oldLeft.getVal ();
    setHeight ();
  }
  
  
  Occupied (Data <T> valIn, AVLNode <T> leftIn, AVLNode <T> rightIn) {
    val = valIn;
    left = leftIn;
    right = rightIn;
    setHeight ();
  }
  
  AVLNode <T> insert (double myKey, T myVal) {
     
    if (myKey <= val.getScore ()) {
      left = left.insert (myKey, myVal); 
    } else {
      right = right.insert (myKey, myVal); 
    }
    
    if (left.getHeight () - right.getHeight () > 1)
      raiseLeft ();
    else if (right.getHeight () - left.getHeight () > 1)
      raiseRight ();
    
    setHeight ();
    return this;
  }
}

class ChrisAVLTree <T> {

  AVLNode <T> root = new EmptyNode <T> ();
  
  void insert (double val, T insertMe) {
    root = root.insert (val, insertMe);
  }
  
  public ArrayList <T> toList () {
    ArrayList <T> temp = new ArrayList <T> ();
    root.toList (temp);
    return temp;
  }
  
  void removeBig () {
    root = root.removeBig ();
  }
  
  public double getBig () {
    return root.getBig ();  
  }
  
  String print () {
    return root.print ();
  }
  
  void checkBalanced () {
    root.checkBalanced ();
  }
} 


// this little class processes a printed version of a BST to see if it is balanced
// enough to be an AVL tree.  We assume that an empty tree is encoded as "()".  And
// we assume that a non-empty tree is encoded as "(<left tree> value <right tree>)".
// So, for example, "(((()0(()1()))2((()3())4(()5())))6())" encodes one tree, and
// "((()0(()1()))2(()3()))" encodes another tree.  This second one has a 2 at the
// root, with a 3 to the right of the root and a 1 to the left of the root.  To the
// left of the one is a 0.  Everything else is empty.
class IsBalanced {
  
  // returns the height of the tree encoded in the string; throws an exception if it
  // is not balanced
  int checkHeight (String checkMe) {
  
    // find the position of the first '('
    int startPos = 0;
    while (checkMe.charAt (startPos) != '(')
      startPos++;
    
    // this is the depth at the left
    int leftDepth = -1;
    
    // and the depth at the right
    int rightDepth = -1;
    
    // now, find where the number of parens on each side is equal
    int lParens = 0;
    int rParens = 0;
    for (int i = startPos + 1; i < checkMe.length (); i++) {
      
      // count each ) or ( in the string
      if (checkMe.charAt (i) == '(')
        lParens++;
      else if (checkMe.charAt (i) == ')')
        rParens++;
      
      // if the number of ) and ( are equal
      if (lParens == rParens && lParens > 0) {
        
        // in this case, we just completed the left tree
        if (leftDepth == -1) {
          leftDepth = checkHeight (checkMe.substring (startPos + 1, i + 1));
          startPos = i + 1;
          lParens = 0;
          rParens = 0;
          
        // in this case, we just completed the right tree
        } else {
          rightDepth = checkHeight (checkMe.substring (startPos + 1, i + 1));
          startPos = i + 1;
          break;
        }
      }
    }
    
    // check to see if this is not a valid AVL tree
    if (leftDepth - rightDepth >= 2 || leftDepth - rightDepth <= -2)
      throw new RuntimeException ("this tree is not balanced! Left: " + leftDepth + " Right: " + rightDepth);
    
    // search for the closing )
    while (checkMe.charAt (startPos) != ')')
      startPos++;
    
    // and we are outta here
    if (leftDepth > rightDepth)
      return leftDepth + 1;
    else
      return rightDepth + 1;
    
  }
  
  
}

class AVLTopKMachine <T> implements ITopKMachine <T> {
 
  private int spaceLeft;
  private ChrisAVLTree <T> myTree = new ChrisAVLTree <T> ();
  private double cutoff = Double.POSITIVE_INFINITY;
  
  public AVLTopKMachine (int kIn) {
    if (kIn < 0)
      throw new RuntimeException ("K must be at least zero.");
    if (kIn == 0)
      cutoff = Double.NEGATIVE_INFINITY;
    spaceLeft = kIn;
  }
  
  public void insert (double score, T value) {
    if (spaceLeft > 0 || score < cutoff) {
      myTree.insert (score, value);
      spaceLeft--;
      if (spaceLeft == 0)
        cutoff = myTree.getBig ();
    }
    
    if (spaceLeft < 0) {
      myTree.removeBig ();
      cutoff = myTree.getBig ();
      spaceLeft++;
    }
      
   
  }
  
  public double getCurrentCutoff () {
    return cutoff; 
  }
  
  public String toString () {
    return myTree.print ();  
  }
  
  public ArrayList <T> getTopK () {
    return myTree.toList (); 
  }
  
  
}


/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
public class TopKTester extends TestCase {
  
  // this simple method randomly shuffles the items in an array
  Random shuffler = new Random (324234);
  void shuffle (Integer [] list) {
    for (Integer i = 0; i < list.length; i++) {
      Integer pos = i + shuffler.nextInt (list.length - i);
      Integer temp = list[i];
      list[i] = list[pos];
      list[pos] = temp;
    }
  }
  
  // the first param is the number of inserts to try.  The second is k. If the third param is true, 
  // then we do a random order of inserts.  If the third param is false, then we do inserts in order,
  // and the method expects a fourth boolean param that tells us whether we do reverse or forward 
  // inserts.  So testInserts (100, 5, false, true) will do 100 reverse order inserts, with a k of 5.
  // testInserts (100, 5, false, false) will do 100 in order inserts.  testInserts (100, 5, true) will
  // do 100 random inserts.
  private void testInserts (int numInserts, int k, boolean... controlTest) {
   
    // see what kind of test to do
    boolean reverseOrNot = false;
    boolean randomOrNot = controlTest[0];
    if (!randomOrNot) 
      reverseOrNot = controlTest[1];
    
    // create a list of random ints
    ITopKMachine <Integer> testMe = new AVLTopKMachine <Integer> (k);
    Integer [] list = new Integer [numInserts];
    for (int i = 0; i < numInserts; i++) {
      if (reverseOrNot)
        list[i] = numInserts - 1 - i;
      else
        list[i] = i; 
    }
    
    // if we are looking for randomness, shuffle the list
    if (randomOrNot)
      shuffle (list);
    
    // now add the ints
    for (int j = 0; j < list.length; j++) {
      
      Integer i = list[j];
      testMe.insert (i * 1.343432, i);
      
      // if we are not random, check to see that the cutoff is correct
      if (!randomOrNot) {
        
        double score = testMe.getCurrentCutoff ();
        
        // if we don't have k inserts, then we should have an extreme cutoff
        if (j + 1 < k) {
          if (score != Double.POSITIVE_INFINITY && score != Double.NEGATIVE_INFINITY)
            fail ("at insert number " + j + 1 + " expected positive or negative infinity; found " + score);
          
        // otherwise, check the cutoff
        } else {
          
          // for reverse order inserts, we have one cutoff...
          if (reverseOrNot) {
            
            assertEquals ("when checking cutoff during reverse inserts", (i + k - 1) * 1.343432, score);
            
          // and for in-order, we have another
          } else {
            assertEquals ("when checking cutoff during in order inserts", (k - 1) * 1.343432, score);
          }
        }
      }
    }
    
    // and make sure top k are correct
    ArrayList <Integer> retVal = testMe.getTopK ();
    Collections.sort (retVal);
    
    // don'e go past the size of the list
    if (k > numInserts)
      k = numInserts;
    
    // make sure the list is the right size
    assertEquals (retVal.size (), k);
    
    // and check its contents
    for (int i = 0; i < k; i++) {
      assertEquals ("when checking values returned getting top k", i, (int) retVal.get (i));
    }
  }
  
  // this checks for balance.. it does NOT check for the right answer... params ae same as above
  private void testBalance (int numInserts, int k, boolean... controlTest) {
       
    // see what kind of test to do
    boolean reverseOrNot = false;
    boolean randomOrNot = controlTest[0];
    if (!randomOrNot) 
      reverseOrNot = controlTest[1];
    
    // create a list of random ints
    ITopKMachine <Integer> testMe = new AVLTopKMachine <Integer> (k);
    Integer [] list = new Integer [numInserts];
    for (int i = 0; i < numInserts; i++) {
      if (reverseOrNot)
        list[i] = numInserts - 1 - i;
      else
        list[i] = i; 
    }
    
    // if we are looking for randomness, shuffle the list
    if (randomOrNot)
      shuffle (list);
    
    // now add the ints
    for (int j = 0; j < list.length; j++) { 
      Integer i = list[j];
      testMe.insert (i * 1.343432, i);
      
      // and check for balance
      if (j % 10 == 0) {
        IsBalanced temp = new IsBalanced ();
        try {
          temp.checkHeight (testMe.toString ());
        } catch (Exception e) {
          fail ("the tree was found to not be balanced"); 
        }
      }
      
      // check for balance one last time
      IsBalanced temp = new IsBalanced ();
      try {
        temp.checkHeight (testMe.toString ());
      } catch (Exception e) {
        fail ("the tree was found to not be balanced"); 
      }
    }      

  }
  
  
  /**
   * A test method.
   * (Replace "X" with a name describing the test.  You may write as
   * many "testSomething" methods in this class as you wish, and each
   * one will be called when running JUnit over this class.)
   */
  
  /******************************
    * These do RANDOM inserts *
    ******************************/
  
  public void testAFewRandomInsertsSmallK () {
    testInserts (10, 5, true);
  }
  
  public void testALotOfRandomInsertsSmallK () {
    testInserts (100, 5, true);
  }
  
  public void testAHugeNumberOfRandomInsertsSmallK () {
    testInserts (100000, 5, true);
  }
  
  public void testAFewRandomInsertsBigK () {
    testInserts (10, 100, true);
  }
  
  public void testALotOfRandomInsertsBigK () {
    testInserts (100, 100, true);
  }
  
  public void testAHugeNumberOfRandomInsertsBigK () {
    testInserts (100000, 100, true);
  }
 
    /******************************
    * These do ORDERED inserts *
    ******************************/
  
  public void testAFewOrderedInsertsSmallK () {
    testInserts (10, 5, false, false);
  }
  
  public void testALotOfOrderedInsertsSmallK () {
    testInserts (100, 5, false, false);
  }
  
  public void testAHugeNumberOfOrderedInsertsSmallK () {
    testInserts (100000, 5, false, false);
  }
  
  public void testAFewOrderedInsertsBigK () {
    testInserts (10, 100, false, false);
  }
  
  public void testALotOfOrderedInsertsBigK () {
    testInserts (100, 100, false, false);
  }
  
  public void testAHugeNumberOfOrderedInsertsBigK () {
    testInserts (100000, 100, false, false);
  }
  
  /******************************
    * These do REVERSE inserts *
    ******************************/
  
  public void testAFewReverseInsertsSmallK () {
    testInserts (10, 5, false, true);
  }
  
  public void testALotOfReverseInsertsSmallK () {
    testInserts (100, 5, false, true);
  }
  
  public void testAHugeNumberOfReverseInsertsSmallK () {
    testInserts (100000, 5, false, true);
  }
  
  public void testAFewReverseInsertsBigK () {
    testInserts (10, 100, false, true);
  }
  
  public void testALotOfReverseInsertsBigK () {
    testInserts (100, 100, false, true);
  }
  
  public void testAHugeNumberOfReverseInsertsBigK () {
    testInserts (100000, 100, false, true);
  }
  
    /***************************
    * Now check for balance!!! *
    **************************/
  
    /******************************
    * These do RANDOM inserts *
    ******************************/
  
  public void testAFewRandomInsertsSmallK_CheckBalance () {
    testBalance (10, 5, true);
  }
  
  public void testALotOfRandomInsertsSmallK_CheckBalance () {
    testBalance (100, 5, true);
  }
  
  public void testAHugeNumberOfRandomInsertsSmallK_CheckBalance () {
    testBalance (100000, 5, true);
  }
  
  public void testAFewRandomInsertsBigK_CheckBalance_CheckBalance () {
    testBalance (10, 100, true);
  }
  
  public void testALotOfRandomInsertsBigK_CheckBalance () {
    testBalance (100, 100, true);
  }
  
  public void testAHugeNumberOfRandomInsertsBigK_CheckBalance () {
    testBalance (100000, 100, true);
  }
 
    /******************************
    * These do ORDERED inserts *
    ******************************/
  
  public void testAFewOrderedInsertsSmallK_CheckBalance () {
    testBalance (10, 5, false, false);
  }
  
  public void testALotOfOrderedInsertsSmallK_CheckBalance () {
    testBalance (100, 5, false, false);
  }
  
  public void testAHugeNumberOfOrderedInsertsSmallK_CheckBalance () {
    testBalance (100000, 5, false, false);
  }
  
  public void testAFewOrderedInsertsBigK_CheckBalance () {
    testBalance (10, 100, false, false);
  }
  
  public void testALotOfOrderedInsertsBigK_CheckBalance () {
    testBalance (100, 100, false, false);
  }
  
  public void testAHugeNumberOfOrderedInsertsBigK_CheckBalance () {
    testBalance (100000, 100, false, false);
  }
  
  /******************************
    * These do REVERSE inserts *
    ******************************/
  
  public void testAFewReverseInsertsSmallK_CheckBalance () {
    testBalance (10, 5, false, true);
  }
  
  public void testALotOfReverseInsertsSmallK_CheckBalance () {
    testBalance (100, 5, false, true);
  }
  
  public void testAHugeNumberOfReverseInsertsSmallK_CheckBalance () {
    testBalance (100000, 5, false, true);
  }
  
  public void testAFewReverseInsertsBigK_CheckBalance () {
    testBalance (10, 100, false, true);
  }
  
  public void testALotOfReverseInsertsBigK_CheckBalance () {
    testBalance (100, 100, false, true);
  }
  
  public void testAHugeNumberOfReverseInsertsBigK_CheckBalance () {
    testBalance (100000, 100, false, true);
  }
  
}