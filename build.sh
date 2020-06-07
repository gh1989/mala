if [ ! -d "../mala_build" ]
then
    mkdir ../mala_build
fi

cd ../mala_build
cmake ../mala
cmake --build .

