import torch


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None


class TraceRays(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pipeline,
        _points,
        _attributes,
        _point_adjacency,
        _point_adjacency_offsets,
        rays,
        start_point,
        depth_quantiles,
        return_contribution,
        #alpha_output,
        # --- SDFOAM --
        use_sdfoam: bool = False,
        inv_s: float = 50.0,
        eps: float = 1e-6,
        sdf_channel: int | None = None,
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.pipeline = pipeline
        ctx.points = _points
        ctx.attributes = _attributes
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets

        ctx.depth_quantiles = depth_quantiles #None
        #ctx.alpha_output = alpha_output

        # Stash NeuS options for backward
        ctx.use_sdfoam = bool(use_sdfoam)
        ctx.inv_s = inv_s
        ctx.eps = float(eps)
        ctx.sdf_channel = (
            int(sdf_channel)
            if (sdf_channel is not None)
            else (_attributes.shape[-1] - 1 if use_sdfoam else -1)
        )

        # Assemble kwargs conditionally
        kwargs = {}
        kwargs.update(dict(
            alpha_mode="sdfoam" if ctx.use_sdfoam else "standard",
            inv_s=ctx.inv_s,           # <-- tensor
            eps=ctx.eps,
            sdf_channel=ctx.sdf_channel,
        ))

        results = pipeline.trace_forward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles=depth_quantiles,             # DISABLED: keep quantiles off
            return_contribution=return_contribution,
            #alpha_output=alpha_output,
            **kwargs,
        )

        ctx.rgba = results["rgba"]
        ctx.depth_indices = results.get("depth_indices", None)

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["rgba"],
            results.get("depth", None),
            results.get("contribution", None),
            results["num_intersections"],
            errbox,
            results["alpha"],
        )

    @staticmethod
    def backward(
        ctx,
        grad_rgba,
        grad_depth,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
        grad_alpha,
    ):
        del grad_contribution, grad_num_intersections, errbox_grad, grad_alpha

        rays = ctx.rays
        start_point = ctx.start_point
        pipeline = ctx.pipeline
        rgba = ctx.rgba
        _points = ctx.points
        _attributes = ctx.attributes
        _point_adjacency = ctx.point_adjacency
        _point_adjacency_offsets = ctx.point_adjacency_offsets
        depth_quantiles = ctx.depth_quantiles

        #alpha_output = ctx.alpha_output

        #--- SDFOAM --
        kwargs = {}
        if ctx.use_sdfoam:
            kwargs.update(
                dict(
                    alpha_mode="sdfoam",
                    inv_s=ctx.inv_s,
                    eps=ctx.eps,
                    sdf_channel=ctx.sdf_channel,
                )
            )
        else:
            kwargs.update(dict(
                alpha_mode="standard",
                inv_s=ctx.inv_s,
                eps=ctx.eps,
                sdf_channel=ctx.sdf_channel,
            ))

        results = pipeline.trace_backward(
            _points,
            _attributes,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            rgba,
            grad_rgba,
            depth_quantiles, #None,  # depth_quantiles
            ctx.depth_indices,# #None,  # depth_indices
            grad_depth, #None,  # grad_depth
            ctx.errbox.ray_error,
            **kwargs,
        )

        points_grad = results["points_grad"].contiguous()
        attr_grad   = results["attr_grad"].contiguous()
        sharpness_grad = results["sharpness_grad"].contiguous()
        if sharpness_grad is not None:
            sharpness_grad = sharpness_grad.to(ctx.inv_s.dtype).view_as(ctx.inv_s)
        ctx.errbox.point_error = results.get("point_error", None)

        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0
        if sharpness_grad is not None:
            sharpness_grad[~sharpness_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.pipeline,
            ctx.rgba,
            ctx.points,
            ctx.attributes,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
            ctx.depth_quantiles,
            #ctx.alpha_output,
        )

        if ctx.use_sdfoam:
            del (ctx.inv_s, ctx.eps, ctx.sdf_channel)

        return (
            None,        # pipeline
            points_grad, # _points
            attr_grad,   # _attributes
            None,        # _point_adjacency
            None,        # _point_adjacency_offsets
            None,        # rays
            None,        # start_point
            None,        # depth_quantiles
            None,        # return_contribution
            None,        # use_sdfoam
            sharpness_grad if ctx.use_sdfoam else None,  # inv_s
            None,        # eps
            None,        # sdf_channel
        )
