import junit.framework.TestCase;
import java.io.*;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.ArrayList;

/**
 * This silly little class wraps String, int pairs so they can be sorted.
 * Used by the WordCounter class.
 */
class WordWrapper implements Comparable <WordWrapper> {

  private String myWord;
  private int myCount;
  private int mySlot;

  /**
   * Creates a new WordWrapper with string "XXX" and count zero.
   */
  public WordWrapper (String myWordIn, int myCountIn, int mySlotIn) {
    myWord = myWordIn;
    myCount = myCountIn;
    mySlot = mySlotIn;
  }
  /**
   * Returns a copy of the string in the WordWrapper.
   *
   * @return    The word contained in this object.  Is copied out: no aliasing.
   */
  public String extractWord () {
    return new String (myWord);
  }

  /**
   * Returns the count in this WordWrapper.
   *
   * @return    The count contained in the object.
   */
  public int extractCount () {
    return myCount;
  }

  public void incCount () {
    myCount++;
  }

  public int getCount () {
    return myCount;
  }

  public int getPos () {
    return mySlot;
  }

  public void setPos (int toWhat) {
      
          mySlot = toWhat;
  }

  public String toString () {
    return myWord + " " + mySlot;
  }

  /**
   * Comparison operator so we implement the Comparable interface.
   *
   * @param  w   The word wrapper to compare to
   * @return     a positive, zero, or negative value depending upon whether this
   *             word wrapper is less then, equal to, or greater than param w.  The
   *             relationship is determied by looking at the counts.
   */
  public int compareTo (WordWrapper w) {
    // Check if the count of the current word (myCount) is equal to the count of the word 'w'
    // If the counts are equal, compare the words lexicographically
    if (w.myCount == myCount) {
      return myWord.compareTo(w.myWord);
    }
    // If the counts are not equal, subtract the count of the current word from the count of the word 'w'
    return w.myCount - myCount;
  }
}


/**
 * This class is used to count occurences of words (strings) in a corpus.  Used by
 * the IndexedDocumentCollection class to find the most frequent words in the corpus.
 */

class WordCounter {
    
  // this is the map that holds (for each word) the number of times we've seen it
  // note that Integer is a wrapper for the buitin Java int type
  private HashMap <String, WordWrapper> wordsIHaveSeen = new HashMap <String, WordWrapper> ();
  private TreeMap <String, String> ones = new TreeMap <String, String> ();
  
  // the set of words that have been extracted
  private ArrayList<WordWrapper> extractedWords = new ArrayList <WordWrapper> ();

  /**
   * Accepts a word and increments the count of the number of times we have seen it.
   * Note that a deep copy of the param is done (no aliasing).
   *
   * @param addMe   The string to count
   */
  public void insert (String addMe) {
    if (ones.containsKey (addMe)) {
      ones.remove (addMe);
      WordWrapper temp = new WordWrapper (addMe, 1, extractedWords.size ());
      wordsIHaveSeen.put(addMe, temp);
      extractedWords.add (temp);
    }

    // first check to see if the word is alredy in wordsIHaveSeen
    if (wordsIHaveSeen.containsKey (addMe)) {
      // if it is, then remove and increment its count
      WordWrapper temp = wordsIHaveSeen.get (addMe);
      temp.incCount ();

      // find the slot that we go to in the extractedWords list
      // by sorting the 'extractedWords' list in ascending order
      for (int i = temp.getPos () - 1; i >= 0 && 
        extractedWords.get (i).compareTo (extractedWords.get (i + 1)) > 0; i--) {
        temp = extractedWords.get (i + 1);
        temp.setPos (i);
        
        WordWrapper temp2 = extractedWords.get (i);
        temp2.setPos (i + 1);
        
        extractedWords.set (i + 1, temp2);
        extractedWords.set (i, temp);
      }

    // in this case it is not there, so just add it
    } else {
      ones.put (addMe, addMe);
    }
  }

  /**
   * Returns the kth most frequent word in the corpus so far.  A deep copy is returned,
   * so no aliasing is possible.  Returns null if there are not enough words.
   *
   * @param k     Note that the most frequent word is at k = 0
   * @return      The kth most frequent word in the corpus to date
   */
  public String getKthMostFrequent (int k) {

    // if we got here, then the array is all set up, so just return the kth most frequent word
    if (k >= extractedWords.size ()) {
      int which = extractedWords.size ();
      for (String s : ones.navigableKeySet ()) {
        if (which == k)
          return s;
        which++;
      }
      return null;
    } else {
      // If k is less than the size of the extractedWords list,
      // return the kth most frequent word from extractedWords
      return extractedWords.get (k).extractWord ();
    }
  }
}
/**
 * A JUnit test case class.
 * Every method starting with the word "test" will be called when running
 * the test with JUnit.
 */
public class CounterTester extends TestCase {

  /**
   * File object to read words from, with buffering.
   */
  private FileReader     file = null;
  private BufferedReader reader = null;

  /**
   * Close file readers.
   */
  private void closeFiles() {
    try {
      if (file != null) {
        file.close();
        file = null;
      }
      if (reader != null) {
        reader.close();
        reader = null;
      }
    } catch (Exception e) {
      System.err.println("Problem closing file");
      System.err.println(e);
      e.printStackTrace();
    }
  }

  /**
   * Close files no matter what happens in the test.
   */
  protected void tearDown () {
    closeFiles();
  }

  /**
   * Open file and set up readers.
   *
   * @param fileName  name of file to open
   *   */
  private void openFile(String fileName) {
    try {
      file = new FileReader(fileName);
      reader = new BufferedReader(file);
    } catch (Exception e) {
      System.err.format("Problem opening %s file\n", fileName);
      System.err.println(e);
      e.printStackTrace();
      fail();
    }
  }

  /**
   * Read the next numWords from the file.
   *
   * @param counter   word counter to update
   * @param numWords  number of words to read.
   */
  private void readWords(WordCounter counter, int numWords) {
    try {
      for (int i=0; i<numWords; i++) {
        if (i % 100000 == 0)
          System.out.print (".");
        String word = reader.readLine();
        if (word == null) {
          return;
        }
        // Insert 'word' into the counter.
        counter.insert(word);
      }
    } catch (Exception e) {
      System.err.println("Problem reading file");
      System.err.println(e);
      e.printStackTrace();
      fail();
    }
  }

  /**
   * Read the next numWords from the file, mixing with queries
   *
   * @param counter   word counter to update
   * @param numWords  number of words to read.
   */
  private void readWordsMixed (WordCounter counter, int numWords) {
    try {
      int j = 0;
      for (int i=0; i<numWords; i++) {
        String word = reader.readLine();
        // If 'word' is null, it means we've reached the end of the file. So, return from the method.
        if (word == null) {
          return;
        }
        counter.insert(word);

        // If the current iteration number is a multiple of 10 and greater than 100,000, perform a query of the kth most frequent word.
        if (i % 10 == 0 && i > 100000) {
          String myStr = counter.getKthMostFrequent(j++);
        }

        // rest j once we get to 100
        if (j == 100)
          j = 0;
      }
    } catch (Exception e) {
      System.err.println("Problem reading file");
      System.err.println(e);
      e.printStackTrace();
      fail();
    }
  }

  /**
   * Check that a sequence of words starts at the "start"th most
   * frequent word.
   *
   * @param counter   word counter to lookup
   * @param start     frequency index to start checking at
   * @param expected  array of expected words that start at that frequency
   */
  private void checkExpected(WordCounter counter, int start, String [] expected) {
    for (int i = 0; i<expected.length; i++) {
      String actual = counter.getKthMostFrequent(start);
      System.out.format("k: %d, expected: %s, actual: %s\n",
                        start, expected[i], actual);
      assertEquals(expected[i], actual);
      start++;
    }
  }

  /**
   * A test method.
   * (Replace "X" with a name describing the test.  You may write as
   * many "testSomething" methods in this class as you wish, and each
   * one will be called when running JUnit over this class.)
   */
  public void testSimple() {
    System.out.println("\nChecking insert");
    WordCounter counter = new WordCounter();
    counter.insert("pizzaz");
    // Insert the word "pizza" into the counter twice
    counter.insert("pizza");
    counter.insert("pizza");
    String [] expected = {"pizza"};
    checkExpected(counter, 0, expected);
  }

  public void testTie() {
    System.out.println("\nChecking tie for 2nd place");
    WordCounter counter = new WordCounter();
    counter.insert("panache");
    counter.insert("pizzaz");
    counter.insert("pizza");
    counter.insert("zebra");
    counter.insert("pizza");
    counter.insert("lion");
    counter.insert("pizzaz");
    counter.insert("panache");
    counter.insert("panache");
    counter.insert("camel");
    // order is important here
    String [] expected = {"pizza", "pizzaz"};
    checkExpected(counter, 1, expected);
  }


  public void testPastTheEnd() {
    System.out.println("\nChecking past the end");
    WordCounter counter = new WordCounter();
    counter.insert("hi");
    counter.insert("hello");
    counter.insert("greetings");
    counter.insert("salutations");
    counter.insert("hi");
    counter.insert("welcome");
    counter.insert("goodbye");
    counter.insert("later");
    counter.insert("hello");
    counter.insert("when");
    counter.insert("hi");
    assertNull(counter.getKthMostFrequent(8));
  }

  public void test100Top5() {
    System.out.println("\nChecking top 5 of 100 words");
    WordCounter counter = new WordCounter();
    // Open the file "allWordsBig"
    openFile("allWordsBig");
    // Read the next 100 words
    readWords(counter, 100);
    String [] expected = {"edu", "comp", "cs", "windows", "cmu"};
    checkExpected(counter, 0, expected);
  }

  public void test300Top5() {
    System.out.println("\nChecking top 5 of first 100 words");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    readWords(counter, 100);
    String [] expected1 = {"edu", "comp", "cs", "windows", "cmu"};
    checkExpected(counter, 0, expected1);

    System.out.println("Adding 100 more words and rechecking top 5");
    readWords(counter, 100);
    String [] expected2 = {"edu", "cmu", "comp", "cs", "state"};
    checkExpected(counter, 0, expected2);

    System.out.println("Adding 100 more words and rechecking top 5");
    readWords(counter, 100);
    String [] expected3 = {"edu", "cmu", "comp", "ohio", "state"};
    checkExpected(counter, 0, expected3);
  }

  public void test300Words14Thru19() {
    System.out.println("\nChecking rank 14 through 19 of 300 words");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    // Read the next 300 words
    readWords(counter, 300);
    String [] expected = {"cantaloupe", "from", "ksu", "okstate", "on", "srv"};
    checkExpected(counter, 14, expected);
  }

  public void test300CorrectNumber() {
    System.out.println("\nChecking correct number of unique words in 300 words");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    readWords(counter, 300);
   /* check the 122th, 123th, and 124th most frequent word in the corpus so far 
    * and check if the result is not null.
    */
    assertNotNull(counter.getKthMostFrequent(122));
    assertNotNull(counter.getKthMostFrequent(123));
    assertNotNull(counter.getKthMostFrequent(124));
    // check the 125th most frequent word in the corpus so far
    // and check if the result is null.
    assertNull(counter.getKthMostFrequent(125));
  }

  public void test300and100() {
    System.out.println("\nChecking top 5 of 100 and 300 words with two counters");
    WordCounter counter1 = new WordCounter();
    openFile("allWordsBig");
    readWords(counter1, 300);
    closeFiles();

    WordCounter counter2 = new WordCounter();
    openFile("allWordsBig");
    readWords(counter2, 100);

    String [] expected1 = {"edu", "cmu", "comp", "ohio", "state"};
    checkExpected(counter1, 0, expected1);

    String [] expected2 = {"edu", "comp", "cs", "windows", "cmu"};
    checkExpected(counter2, 0, expected2);
  }

  public void testAllTop15() {
    System.out.println("\nChecking top 15 of all words");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    // Read the next 6000000 words
    readWords(counter, 6000000);
    String [] expected = {"the", "edu", "to", "of", "and",
                          "in", "is", "ax", "that", "it",
                          "cmu", "for", "com", "you", "cs"};
    checkExpected(counter, 0, expected);
  }

  public void testAllTop10000() {
    System.out.println("\nChecking time to get top 10000 of all words");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    readWords(counter, 6000000);

    for (int i=0; i<10000; i++) {
      counter.getKthMostFrequent(i);
    }
  }

  public void testSpeed() {
    System.out.println("\nMixing adding data with finding top k");
    WordCounter counter = new WordCounter();
    openFile("allWordsBig");
    readWordsMixed (counter, 6000000);
    String [] expected = {"the", "edu", "to", "of", "and",
                          "in", "is", "ax", "that", "it",
                          "cmu", "for", "com", "you", "cs"};
    checkExpected(counter, 0, expected);
  }

}