Build:
make

Run serial baseline:
./serial_face dataset/high_load
./serial_face dataset/low_load

Run current submitted parallel version:
mpirun -np 12 ./parallel_face dataset/high_load
mpirun -np 12 ./parallel_face dataset/low_load

If using the submitted .out executable files:
mpirun -np 12 ./parallel_face.out dataset/high_load
mpirun -np 12 ./parallel_face.out dataset/low_load

Required files:（檔案太大，因此未附上）
* haarcascade_frontalface_default.xml
* dataset/high_load/
* dataset/low_load/

Note:
The submitted parallel_face.cpp is the final Controlled Pure MPI version.
Therefore, the current source code can reproduce the Serial Baseline and the Controlled Pure MPI version.

Other experimental versions, including Pure MPI Dynamic, Computing Master,
Adaptive Master, and Hybrid MPI + OpenMP, were tested during development.
Their source code versions were overwritten during iteration, so their results are preserved in the submitted log files.

Related logs:

* benchmark_1318056.log: Pure MPI Dynamic results
* final_1318114.log: Computing Master results
* final_1318139.log: Adaptive Master results
* final_1318213.log: Controlled Pure MPI results
* hybrid_1318076.log: Hybrid MPI + OpenMP results
