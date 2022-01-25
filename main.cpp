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

void loadModel()
{
    Graph = TF_NewGraph();
    Status = TF_NewStatus();

    SessionOpts = TF_NewSessionOptions();
    TF_Buffer *RunOpts = NULL;

    const char *saved_model_dir = "mnist_saved_model/"; // Path of the model
    const char *tags = "serve";                         // default model serving tag; can change in future
    int ntags = 1;

    Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    if (TF_GetCode(Status) == TF_OK)
        printf("TF_LoadSessionFromSavedModel OK\n");
    else
        printf("%s", TF_Message(Status));

    //****** Get input tensor
    Input = (TF_Output *)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if (t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

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

void pipeline(cv::Mat img)
{
    float data[INPUT_WIDTH][INPUT_HEIGHT];
    for (int y = 0; y < INPUT_HEIGHT; y++)
    {
        for (int x = 0; x < INPUT_WIDTH; x++)
        {
            data[x][y] = static_cast<float>(img.at<float>(y, x));
        }
    }

    //********* Allocate data for inputs & outputs
    TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumInputs);
    TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumOutputs);

    int ndims = 3;
    int64_t dims[] = {1, INPUT_WIDTH, INPUT_HEIGHT};
    int ndata = sizeof(float) * INPUT_WIDTH * INPUT_HEIGHT; // This is tricky, it number of bytes not number of element

    TF_Tensor *in_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (in_tensor != NULL)
        printf("TF_NewTensor is OK\n");
    else
        printf("ERROR: Failed TF_NewTensor\n");

    InputValues[0] = in_tensor;

    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

    if (TF_GetCode(Status) == TF_OK)
    {
        printf("Session is OK\n");
    }
    else
    {
        printf("%s", TF_Message(Status));
    }

    // //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);
}

int main()
{
    printf("+++\n");

    char *file = "../images/six.jpg";
    cv::Mat img = cv::imread(file, 1);

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::resize(img, img, cv::Size(28, 28), 0, 0, cv::INTER_LINEAR);
    printf("%d\n", img.channels());

    // cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Preview", img);
    // cv::waitKey(0);

    loadModel();
    pipeline(img);

    printf("---\n");
    return 0;
}