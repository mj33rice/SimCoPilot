package dependency_analyzer;

public class ClassExistenceChecker {

    /**
     * Checks if a given subclass/interface exists within a base package.
     *
     * @param basePackage The base package name, without a wildcard. For example, "java.util".
     * @param potentialSubclass The name of the potential subclass or interface. For example, "Map".
     * @return true if the subclass/interface exists within the base package, false otherwise.
     */
    public static boolean doesClassExist(String basePackage, String potentialSubclass) {
        try {
            // Construct the fully qualified name of the potential subclass/interface
            String fullyQualifiedName = basePackage + "." + potentialSubclass;
            
            // Attempt to load the class with the fully qualified name
            Class.forName(fullyQualifiedName);
            
            // If no exception is thrown, the class/interface exists
            return true;
        } catch (ClassNotFoundException e) {
            // If a ClassNotFoundException is caught, the class/interface does not exist
            return false;
        }
    }
    
    // public static void main(String[] args) {
    //     // Example usage
    //     System.out.println(doesClassExist("java.util", "Map")); // Should print true
    //     System.out.println(doesClassExist("java.util", "NonExistentClass")); // Should print false
    // }
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java ClassExistenceChecker <basePackage> <potentialSubclass>");
            return;
        }
        String basePackage = args[0];
        String potentialSubclass = args[1];
        System.out.println(doesClassExist(basePackage, potentialSubclass));
    }
    
}