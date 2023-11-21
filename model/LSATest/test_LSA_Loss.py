import torch
from LSALoss import get_all_LSA_fns,get_loss_fns


def test_LSA_loss_transpose():
    # Define a sample input
    x = torch.rand(10, 10)

    # Get all LSA loss functions
    loss_fns = get_loss_fns()
    failures=[]
    for loss_name, loss_fn in loss_fns.items():
        # Compute the loss for x
        LSA=get_all_LSA_fns()["LSAstock"](x)
        loss_x, logits = loss_fn(LSA,x)

        # Compute the loss for the transpose of x
        loss_x_transpose,logitst = loss_fn(LSA.t(),x.t())

        # Assert that the loss for x and its transpose are the same
        if not torch.all(torch.isclose(loss_x, loss_x_transpose)):
            failures.append("Failed for {}".format(loss_name))
        if not torch.all(torch.isclose(logits, logitst.t())):
            failures.append("Failed for {} logits".format(loss_name))
    if len(failures)==0:
        print("Passed for all loss functions")
    else:
        print("Failed for the following loss functions:")

        for failure in failures:
            print(failure)


if __name__ == "__main__":
    test_LSA_loss_transpose()
