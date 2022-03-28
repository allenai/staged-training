import torch
from transformers import AutoModelForCausalLM
from transformers.optimization import AdamW

from gpt_pretrain import (double_matrix_weight, double_param,
                          double_split_matrix_weight, double_weights)


def test_double_matrix_weight():
    weight = torch.rand(52, 88)
    x = torch.rand(88, 1)
    weight2 = double_matrix_weight(
        weight,
        is_grad=False,
        is_avg_sq=False,
    )
    y = torch.matmul(weight, x)
    y2 = torch.matmul(weight2, torch.cat([x, x], dim=0))
    assert torch.allclose(y, y2[:52], atol=1e-05, rtol=1e-03)

    x = torch.rand(1, 11)
    c_attn = torch.rand(11, 11 * 3)
    y0, y1, y2 = torch.matmul(x, c_attn).split(11, dim=1)

    c_attn2 = double_split_matrix_weight(
        c_attn,
        is_grad=False,
        is_avg_sq=False,
    )

    y00, y11, y22 = torch.matmul(torch.cat([x, x], dim=1), c_attn2).split(11 * 2, dim=1)

    assert torch.allclose(y0, y00[:, :11], atol=1e-05, rtol=1e-03)


def test_double_gradients():

    # model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    optimizer = AdamW(model.parameters(), lr=0.00000, betas=(0.0, 0.0))
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=input_ids, labels=labels)[0]
    loss.backward()
    optimizer.step()
    # model.roberta.embeddings.word_embeddings.weight.grad
    double_model = double_weights(
        model,
        is_double_embedding=True,
    )
    double_optimizer = AdamW(double_model.parameters(), lr=0.00000, betas=(0.0, 0.0))
    double_model.eval()
    double_loss = double_model(input_ids=input_ids, labels=labels)[0]
    double_loss.backward()
    double_optimizer.step()

    assert torch.allclose(double_loss, loss, atol=1e-05, rtol=1e-03)
    for (
        (name, parameter),
        (double_name, double_parameter),
        (opt_key, opt_val),
        (double_opt_key, double_opt_val),
    ) in zip(
        model.named_parameters(),
        double_model.named_parameters(),
        optimizer.state.items(),
        double_optimizer.state.items(),
    ):
        assert name == double_name
        assert id(parameter) == id(opt_key)
        assert id(double_parameter) == id(double_opt_key)
        predicted = double_param(
            name,
            parameter.grad,
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=False,
        )
        assert torch.allclose(predicted, double_parameter.grad, atol=1e-05, rtol=1e-03)

        predicted = double_param(
            name,
            opt_val["exp_avg"],
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=False,
        )
        assert torch.allclose(
            predicted, double_opt_val["exp_avg"], atol=1e-05, rtol=1e-03
        )

        predicted = double_param(
            name,
            opt_val["exp_avg_sq"],
            is_double_embedding=True,
            is_grad=True,
            is_avg_sq=True,
        )
        assert torch.allclose(
            predicted, double_opt_val["exp_avg_sq"], atol=1e-05, rtol=1e-03
        )
