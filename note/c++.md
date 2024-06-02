# C++

## 头文件

C++支持声明与定义的分离。如果有一个很常用的函数 `void f()`，在整个程序中的许多 .cpp 文件中都会被调用，那么，只需要在一个文件中**定义**这个函数，而在其他的文件中**声明**这个函数就可以了。`#include`的作用是把它后面所写的那个文件的内容，完完整整地、一字不改地包含到当前的文件中来。通过将声明放在头文件中，编译器只需要在头文件更改时重新编译相关的源文件，从而提高了编译速度。因此，**头文件中应该只放变量和函数的声明，而不能放它们的定义**。但也存在一些例外：

* 头文件中可以写**内联函数（inline）**的定义。因为inline函数是需要编译器在遇到它的地方根据它的定义把它内联展开的。
* 对于类，一般的做法是把**类的定义**放在头文件中，而把函数成员的实现代码放在一个 .cpp 文件中。不过，还有另一种办法。那就是直接把函数成员的实现代码也写进类定义里面。在 C++ 的类中，如果函数成员在类的定义体中被定义，那么编译器会视这个函数为内联的。因此，把函数成员的定义写进类定义体，一起放进头文件中，是合法的。但头文件中不能包含static 成员函数和成员变量的定义，只能声明。

系统自带的头文件用尖括号括起来，用户自定义的文件用双引号括起来。编译器首先会在用户目录下查找，然后在到 C++ 安装目录（比如 VC 中可以指定和修改库文件查找路径，Unix 和 Linux 中可以通过环境变量来设定）中查找，最后在系统文件中查找。

## 编译链接

* **编译：**源代码转换成目标文件（`.cpp` -> `.o`）
* **链接：**多个目标文件合并成最后的可执行文件

**g++使用：**

```bash
g++ -o target_name source1.cpp source2.cpp -I header_path -L libfile_path
```

## CMake

CMake是一种跨平台编译工具，CMake主要是编写CMakeLists.txt文件，然后用cmake命令将CMakeLists.txt文件转化为make所需要的Makefile文件。

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProgram)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含当前目录中的头文件
include_directories(/usr/local/include)

# 指定库文件路径
link_directories(/usr/local/lib)

# 指定源文件
set(SOURCES
    main.cpp
)

# 添加可执行文件
add_executable(my_program ${SOURCES})

# 链接第三方库
target_link_libraries(my_program foo)
```
