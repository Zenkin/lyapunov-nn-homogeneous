import torch
import torch.nn.functional as F


def loss_L2_article(model, xt, f_tilde):
    """
    Реализация формулы (10) из статьи.
    model — WArticle
    xt — точки (requires_grad=True)
    f_tilde — f(x) в тех же точках
    """

    # w = T(x)
    w = model.T(xt)  # (N,2)

    # ----- вычисляем p f(x) = DT(x) @ f(x) -----
    jv_parts = []

    for k in range(w.shape[1]):
        Tk = w[:, k]

        grad_Tk = torch.autograd.grad(
            outputs=Tk.sum(),
            inputs=xt,
            create_graph=True,
            retain_graph=True,
        )[0]  # (N,2)

        jv_k = (grad_Tk * f_tilde).sum(dim=1)  # (N,)
        jv_parts.append(jv_k)

    jv = torch.stack(jv_parts, dim=1)  # (N,2)

    # ----- собираем формулу -----

    w2 = (w * w).sum(dim=1)  # w^T w
    wf = (w * jv).sum(dim=1)  # w^T (DT f)

    term = 2.0 * wf + w2  # 2 w^T p f + w^T w

    loss_decay = F.relu(term)
    loss_pos = F.relu(-w2)

    return (loss_decay + loss_pos).mean()

def scalar_lie_derivative(values, inputs, vector_field, create_graph: bool):
    """
    Считает скалярную производную Ли для батча:

        dV = grad(V) · f

    values:
        либо (N,), либо (N,1)
    inputs:
        тензор координат формы (N,d), по которому берётся градиент
    vector_field:
        поле f(x) той же формы (N,d)

    Возвращает:
        dV формы (N,)
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape (N,d), got {tuple(inputs.shape)}")

    if vector_field.shape != inputs.shape:
        raise ValueError(
            f"vector_field must have the same shape as inputs, got {tuple(vector_field.shape)} vs {tuple(inputs.shape)}"
        )

    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError(f"values must have shape (N,) or (N,1), got {tuple(values.shape)}")
        values_flat = values[:, 0]
    elif values.ndim == 1:
        values_flat = values
    else:
        raise ValueError(f"values must have shape (N,) or (N,1), got {tuple(values.shape)}")

    if values_flat.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"batch mismatch: values has {values_flat.shape[0]} samples, inputs has {inputs.shape[0]}"
        )

    grad_values = torch.autograd.grad(
        outputs=values_flat.sum(),
        inputs=inputs,
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    dV = (grad_values * vector_field).sum(dim=1)
    return dV



def loss_homogeneous_sphere(model, y, f_inf_y):
    """
    Loss для homogeneous-stage на сфере S_r(1).

    Обучающие условия:
        V(y) >= 1
        dV_inf(y) <= -1

    где
        dV_inf(y) = grad(V(y)) · f_inf(y)

    Возвращает словарь метрик. Ключ "loss" содержит тензор,
    пригодный для backward(). Остальные значения — scalar tensor-метрики.
    """
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"y must have shape (N,2), got {tuple(y.shape)}")

    if f_inf_y.shape != y.shape:
        raise ValueError(
            f"f_inf_y must have the same shape as y, got {tuple(f_inf_y.shape)} vs {tuple(y.shape)}"
        )

    # Делаем y leaf-тензором с градиентом: loss зависит от параметров модели,
    # а производная Ли требует grad по координатам.
    y_req = y.detach().clone().requires_grad_(True)
    f_inf_req = f_inf_y.detach().clone()

    V = model.sphere_forward(y_req)
    dV_inf = scalar_lie_derivative(
        values=V,
        inputs=y_req,
        vector_field=f_inf_req,
        create_graph=True,
    )

    V_flat = V[:, 0]

    loss_decay_vec = F.relu(dV_inf + 1.0)
    loss_pos_vec = F.relu(1.0 - V_flat)

    loss_decay = loss_decay_vec.mean()
    loss_pos = loss_pos_vec.mean()
    loss = loss_decay + loss_pos

    with torch.no_grad():
        metrics = {
            "loss": loss,
            "loss_decay": loss_decay.detach(),
            "loss_pos": loss_pos.detach(),
            "v_min": V_flat.min().detach(),
            "v_mean": V_flat.mean().detach(),
            "dV_inf_max": dV_inf.max().detach(),
            "viol_decay_rate": (loss_decay_vec > 0.0).to(y.dtype).mean().detach(),
            "viol_pos_rate": (loss_pos_vec > 0.0).to(y.dtype).mean().detach(),
        }

    return metrics
