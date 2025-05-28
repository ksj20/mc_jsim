main.cpp is the base (golden rule) code

main1.cpp adds spaces between cout, string, and endl

--- main.cpp    2025-05-16 17:14:38.614071177 +0900
+++ main1.cpp   2025-05-16 17:17:41.872749668 +0900
@@ -2,6 +2,6 @@
 
 int main() {
     using namespace std;
-    cout << "Hello" << endl;
+    cout  <<  "Hello"  <<  endl;
     return 0;
 }

main2.cpp changes the position of using namespace std

--- main.cpp    2025-05-16 17:14:38.614071177 +0900
+++ main2.cpp   2025-05-16 17:16:17.422199103 +0900
@@ -1,7 +1,7 @@
 #include <iostream>
+using namespace std;
 
 int main() {
-    using namespace std;
     cout << "Hello" << endl;
     return 0;
 }

main3.cpp divides the string Hello into two parts: "Hel" and "lo"

--- main.cpp    2025-05-16 17:14:38.614071177 +0900
+++ main3.cpp   2025-05-16 17:17:03.391402786 +0900
@@ -2,6 +2,7 @@
 
 int main() {
     using namespace std;
-    cout << "Hello" << endl;
+    cout << "Hel"<< endl;
+    cout << "lo" << endl;
     return 0;
 }



python similarity.py ../src/main.cpp ../src/main1.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main1.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
100.00% similarity

python similarity.py ../src/main.cpp ../src/main2.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main2.cpp:
      #include <iostream> using namespace std; int main() { cout << "Hello" << endl; return 0; }
85.56% similarity

python similarity.py ../src/main.cpp ../src/main3.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main3.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hel"; cout << "lo" << endl; return 0; }
93.75% similarity

python similarity.py ../src/main.cpp ../src/main4.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main4.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hel" << endl; cout << "lo" << endl; return 0; }
90.00% similarity

python similarity.py ../src/main.cpp ../src/main5.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main5.cpp:
      #include <iostream> int main() { using namespace std; cout << "He"; cout << "l"; cout << "l"; cout << "o" << endl; return 0; }
83.33% similarity

python similarity.py ../src/main.cpp ../src/main6.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main6.cpp:
      #include <iostream> int main() { using namespace std; cout << "H"<< endl; cout << "e"<< endl; cout << "l"<< endl; cout << "l"<< endl; cout << "o" << endl; return 0; }
70.31% similarity

python similarity.py ../src/main.cpp ../src/main7.cpp true
../src/main.cpp:
      #include <iostream> int main() { using namespace std; cout << "Hello" << endl; return 0; }
../src/main7.cpp:
      #include <iostream> #include <cmath> int main() { using namespace std; cout << "Hello" << endl; return 0; }
91.37% similarity