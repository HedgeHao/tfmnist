project(tf CXX)

include_directories("/home/hedgehao/bin/prebuilt/tensorflow/linux/2.6.0/include")
link_directories("/home/hedgehao/bin/prebuilt/tensorflow/linux/2.6.0/lib")

add_executable( ${PROJECT_NAME}
  main.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} tensorflow ${OpenCV_LIBS})
