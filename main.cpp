#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/c/c_api.h"

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28

void NoOpDeallocator(void *data, size_t a, void *b) {}

TF_Graph *Graph;
TF_Session *Session;
TF_SessionOptions *SessionOpts;
TF_Output *Input;
TF_Output *Output;
TF_Status *Status;
int NumInputs = 1;
int NumOutputs = 1;

int loadModel(const char *saved_model_dir, const char *tags)
{
    Graph = TF_NewGraph();
    Status = TF_NewStatus();

    SessionOpts = TF_NewSessionOptions();
    TF_Buffer *RunOpts = NULL;

    int ntags = 1;

    Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if (!TF_GetCode(Status) == TF_OK)
    {
        printf("%s", TF_Message(Status));
        return -1;
    }

    //****** Get input tensor
    Input = (TF_Output *)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_flatten_input"), 0};
    if (t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName\n");
    else
        printf("TF_GraphOperationByName is OK\n");

    Input[0] = t0;

    //********* Get Output tensor
    Output = (TF_Output *)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if (t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
        printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    Output[0] = t2;
}

void pipeline(cv::Mat img, float *output)
{
    //********* Allocate data for inputs & outputs
    TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumInputs);
    TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumOutputs);

    int ndims = 3;
    int64_t dims[] = {1, INPUT_WIDTH, INPUT_HEIGHT};
    int ndata = sizeof(float) * INPUT_WIDTH * INPUT_HEIGHT; // This is tricky, it number of bytes not number of element

    TF_Tensor *in_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, img.data, ndata, &NoOpDeallocator, 0);

    if (in_tensor != NULL)
        printf("TF_NewTensor is OK\n");
    else
        printf("ERROR: Failed TF_NewTensor\n");

    InputValues[0] = in_tensor;

    // Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);
    if (TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s", TF_Message(Status));
    }

    // Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    float *buff = (float *)TF_TensorData(OutputValues[0]);
    for (int i = 0; i < 10; i++)
        output[i] = buff[i];
}

void softmax(float *input, size_t size)
{

    assert(0 <= size <= sizeof(input) / sizeof(double));

    int i;
    double m, sum, constant;

    m = -INFINITY;
    for (i = 0; i < size; ++i)
    {
        if (m < input[i])
        {
            m = input[i];
        }
    }

    sum = 0.0;
    for (i = 0; i < size; ++i)
    {
        sum += exp(input[i] - m);
    }

    constant = m + log(sum);
    for (i = 0; i < size; ++i)
    {
        input[i] = exp(input[i] - constant);
    }
}

int main()
{
    printf("+++\n");

    char *file = "../images/three_hand.png";
    cv::Mat img = cv::imread(file, 1);

    // cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
    img.convertTo(img, CV_32FC1);
    img = img / 255;

    // cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Preview", img);
    // cv::waitKey(0);

    loadModel("mnist_saved_model", "serve");
    float *output = new float[10];
    pipeline(img, output);
    softmax(output, 10);

    int max = 0;
    for (int i = 0; i < 10; i++)
    {
        if (output[i] > output[max])
            max = i;
    }
    printf("result: %d\n", max);

    printf("---\n");
    return 0;
}