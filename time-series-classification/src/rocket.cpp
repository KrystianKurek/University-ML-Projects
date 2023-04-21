#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
    return x * 2;
}
// [[Rcpp::export]]
NumericVector apply_kernel(NumericVector X, NumericVector weights, int length, float bias, int dilation, int padding){
    int input_length = X.size();

    int output_length = (input_length + (2 * padding)) - ((length - 1) * dilation);

    int _ppv = 0;
    float _max = -99999999;

    int end = (input_length + padding) - ((length - 1) * dilation);

    for (int i=-padding; i < end; i++){

        float _sum = bias;

        int index = i;
        for(int j=0; j < length; j++){
            if ((index > -1) && (index < input_length)){
                _sum = _sum + weights[j] * X[index];
            }
            index = index + dilation;
        }

        if (_sum > _max){
            _max = _sum;
        }

        if (_sum > 0){
            _ppv += 1;
        }
    }
    NumericVector result = { _ppv / (double)output_length, _max};
    return result;
}

// [[Rcpp::export]]
std::vector<std::vector<double>> apply_kernels(NumericMatrix X, NumericVector kernels_data, int num_kernels){

    int num_examples = X.nrow();


    int kernels_data_size = kernels_data.size();
    NumericVector weights = kernels_data[Range(1-1,(kernels_data_size-4*num_kernels - 1))];
    NumericVector lengths = kernels_data[Range((kernels_data_size-4*num_kernels + 1 -1),(kernels_data_size-3*num_kernels -1))];
    NumericVector biases = kernels_data[Range((kernels_data_size-3*num_kernels + 1 - 1),(kernels_data_size-2*num_kernels - 1))];
    NumericVector dilations = kernels_data[Range((kernels_data_size-2*num_kernels + 1-1),(kernels_data_size-num_kernels-1))];
    NumericVector paddings = kernels_data[Range((kernels_data_size-num_kernels+1-1),kernels_data_size-1)];


    std::vector<std::vector<double>> _X(num_examples, std::vector<double>(2*num_kernels));

    for(int i=0; i<num_examples; i++){
        int a1 = 0; // for weights
        int a2 = 0; // for features

        for(int j=0; j<num_kernels; j++){

            int b1 = a1 + lengths[j];
            int b2 = a2 + 2;
            NumericVector applied = apply_kernel(X(i,_), weights[Range(a1,b1)], lengths[j], biases[j], dilations[j], paddings[j]);

            _X[i][2*j] = applied[0];
            _X[i][2*j+1] = applied[1];



            a1 = b1;
            a2 = b2;
        }
    }
    return _X;