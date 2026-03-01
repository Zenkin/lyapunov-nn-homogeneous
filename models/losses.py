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
