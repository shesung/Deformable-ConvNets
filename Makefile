all:
	mkdir -p ./data
	mkdir -p ./output
	mkdir -p ./model/pretrained_model

	pushd lib/nms/; python setup_linux.py build_ext --inplace; pushd
	pushd lib/bbox/; python setup_linux.py build_ext --inplace; pushd
clean:
	pushd lib/nms/; rm *.so *.c *.cpp; pushd
	pushd lib/bbox/; rm *.so *.c *.cpp; pushd
