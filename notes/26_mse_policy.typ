#let dark = true
#let fg = if dark { rgb("#e6e6e6") } else { black }
#let bg = if dark { rgb("#1b1b1f") } else { white }
#set page(width: 20cm, height: auto, margin: 2cm, fill: bg)
#set text(size: 12pt)
#set math.mat(delim: "[")
#set math.vec(delim: "[")
#let user-color = if dark { rgb("#7aa2ff") } else { rgb("#1a56db") }
#let claude(body) = text(fill: fg, body)
#set text(fill: user-color)
#import "@preview/lilaq:0.5.0" as lq
#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/fletcher:0.5.8": diagram, node, edge
#show: lq.set-spine(stroke: fg)
#show: lq.set-tick(stroke: fg)
#show: lq.set-grid(stroke: 0.5pt + (if dark { rgb("#ffffff30") } else { rgb("#00000020") }))
#show: lq.set-legend(fill: bg, stroke: 0.5pt + fg)
#show: lq.set-diagram(fill: bg)

#let api(s) = highlight(fill: rgb("#7aa2ff40"), raw(s))

#claude[ // claude:begin
#let c-new = rgb("#8f6fb5")
#let c-warn = rgb("#b5655f")
#let c-flag = rgb("#c9a45c")
#let c-dim = rgb("#6b6b73")
#let note(c, title, body) = block(width: 100%, fill: c.transparentize(85%), stroke: 0.6pt + c, inset: 8pt, radius: 3pt)[#text(weight: "bold", fill: fg)[#title] \ #body]
#show raw: set text(size: 8.5pt)
#let t8(s) = text(size: 8pt, fill: fg, s)
#let box-node(pos, label, c, name) = node(pos, text(fill: fg, size: 8.5pt, label), name: name, fill: c.transparentize(75%), stroke: 0.6pt + c, corner-radius: 3pt, inset: 6pt)
#let cell(label, c, w: 0.8cm) = box(width: w, height: 0.65cm, fill: c.transparentize(70%), stroke: 0.6pt + c, radius: 2pt, align(center + horizon, text(fill: fg, size: 8pt, label)))
#let strip(..cs) = stack(dir: ltr, spacing: 2pt, ..cs.pos())

#align(center, block(fill: fg.transparentize(88%), inset: 10pt, radius: 3pt, text(size: 13pt, raw("class MSEPolicy(nn.Module)"))))

= Step 6b: `MSEPolicy`

== 1. Forward: shapes

#align(center)[
  #diagram(
    spacing: (2.05cm, 1.6cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [obs \ raw], c-dim, <a>),
    box-node((1, 0), [`norm_obs`], c-new, <b>),
    box-node((2, 0), [`net`], c-new, <c>),
    box-node((3, 0), [reshape], c-new, <d>),
    box-node((4, 0), [`unnorm_act`], c-new, <e>),
    box-node((5, 0), [`qdot` \ raw], c-dim, <f>),
    edge(<a>, <b>, "->", t8[`(B,7)`], label-side: left),
    edge(<b>, <c>, "->", t8[`(B,7)`], label-side: left),
    edge(<c>, <d>, "->", t8[`(B,48)`], label-side: left),
    edge(<d>, <e>, "->", t8[`(B,16,3)`], label-side: left),
    edge(<e>, <f>, "->", t8[`(B,16,3)`], label-side: left),
  )
]

Training stops after reshape (normalized units). `unnorm_act` is for rollout only.

== 2. The 48 numbers become `(16, 3)`

#align(center)[
  #grid(
    columns: (auto, auto), column-gutter: 16pt, row-gutter: 14pt, align: horizon + left,
    t8[`net` output, `(48,)`],
    strip(cell("0", c-dim), cell("1", c-dim), cell("2", c-dim), box(width: 0.3cm), cell("3", c-flag), cell("4", c-flag), cell("5", c-flag), box(width: 0.3cm), text(fill: fg)[...], box(width: 0.3cm), cell("45", c-dim), cell("46", c-dim), cell("47", c-dim)),
    t8[reshape `(16, 3)`],
    grid(
      columns: (1.4cm,) + (0.8cm,) * 3, column-gutter: 2pt, row-gutter: 2pt, align: center + horizon,
      [], t8[$q dot_0$], t8[$q dot_1$], t8[$q dot_2$],
      t8[action 0], cell("0", c-dim), cell("1", c-dim), cell("2", c-dim),
      t8[action 1], cell("3", c-flag), cell("4", c-flag), cell("5", c-flag),
      t8[...], [], [], [],
      t8[action 15], cell("45", c-dim), cell("46", c-dim), cell("47", c-dim),
    ),
  )
]

== 3. The loss, with the pad mask

#align(center)[
  #diagram(
    spacing: (1.25cm, 1.4cm),
    node-stroke: 0.6pt + fg, edge-stroke: 0.6pt + fg,
    box-node((0, 0), [`pred`], c-new, <p>),
    box-node((0, 1), [target \ `norm_act(chunk)`], c-dim, <t>),
    box-node((1, 0.5), [$("pred" - "target")^2$], c-new, <sq>),
    box-node((2, 0.5), [mean over joints], c-new, <mj>),
    box-node((3, 0.5), [$times$ `keep`], c-new, <mul>),
    box-node((4, 0.5), [loss = sum \ $div$ `keep.sum()`], c-flag, <div>),
    box-node((2, 1.7), [`pad`], c-dim, <pad>),
    box-node((3, 1.7), [`keep = ~pad`], c-new, <keep>),
    edge(<p>, <sq>, "->", t8[`(B,16,3)`], label-side: left),
    edge(<t>, <sq>, "->", t8[`(B,16,3)`], label-side: left),
    edge(<sq>, <mj>, "->", t8[`(B,16,3)`], label-side: left),
    edge(<mj>, <mul>, "->", t8[`(B,16)`], label-side: left),
    edge(<pad>, <keep>, "->", t8[`(B,16)`], label-side: left),
    edge(<keep>, <mul>, "->", t8[`(B,16)`], label-side: left),
    edge(<mul>, <div>, "->", t8[`(B,16)`], label-side: left),
  )
]

== 4. What is inside the object

#block(width: 100%, stroke: 0.6pt + fg, inset: 10pt, radius: 4pt)[
  #text(fill: fg, weight: "bold")[`MSEPolicy(nn.Module)`]
  #v(4pt)
  #grid(
    columns: (1fr, 1fr), column-gutter: 10pt,
    block(fill: c-new.transparentize(80%), stroke: 0.6pt + c-new, inset: 8pt, radius: 3pt, width: 100%, text(fill: fg, size: 9pt)[
      `self.norm` : `Normalizer` \
      #t8[buffers (saved, not trained):] \
      #t8[`obs_mean (7,)` `obs_std (7,)`] \
      #t8[`act_mean (3,)` `act_std (3,)`] \
      #t8[`norm.fit(demos)` once, before training]
    ]),
    block(fill: c-dim.transparentize(80%), stroke: 0.6pt + c-dim, inset: 8pt, radius: 3pt, width: 100%, text(fill: fg, size: 9pt)[
      `self.net` : `nn.Sequential` \
      #t8[parameters (saved, trained):] \
      #t8[`Linear 7 -> H`, ReLU] \
      #t8[`Linear H -> H`, ReLU] \
      #t8[`Linear H -> 48` \  `H` is your choice (256 is typical)]
    ]),
  )
  #v(4pt)
  #text(fill: fg, size: 9pt)[
    `__init__(self)` builds both \
    `loss(self, obs, chunk, pad)` $->$ scalar, takes raw `obs` and `chunk` from `sample` \
    `act(self, obs)` $->$ `(B, 16, 3)` real units
  ]
  #v(2pt)
  #t8[`state_dict()` of the policy holds the net's weights and the four stats together.]
]

== 5. What MSE is expected to learn at the start

#align(center)[
  #canvas(length: 1cm, {
    import draw: *
    set-style(stroke: fg)
    circle((0, 0), radius: 0.9, fill: c-dim.transparentize(60%), stroke: 0.8pt + c-dim)
    content((0, 0), text(fill: fg, size: 8pt)[pillar])
    circle((-5.5, 0), radius: 0.12, fill: fg)
    content((-5.5, -0.6), text(fill: fg, size: 8pt)[start])
    circle((5.5, 0), radius: 0.12, fill: fg)
    content((5.5, -0.6), text(fill: fg, size: 8pt)[goal])
    bezier((-5.5, 0), (5.5, 0), (-2, 3), (2, 3), stroke: 1.2pt + c-new)
    bezier((-5.5, 0), (5.5, 0), (-2, -3), (2, -3), stroke: 1.2pt + c-new)
    content((0, 2.45), text(fill: fg, size: 8pt)[expert, coin = left])
    content((0, -2.45), text(fill: fg, size: 8pt)[expert, coin = right])
    line((-5.5, 0), (-0.9, 0), stroke: (paint: c-warn, thickness: 1.4pt, dash: "dashed"))
    line((0.9, 0), (5.5, 0), stroke: (paint: c-warn, thickness: 1.4pt, dash: "dashed"))
    content((-3.3, 0.45), text(fill: fg, size: 8pt)[average of the two])
  })
]

#align(center, t8[sketch. The obs at the start is the same for both coins, since the coin is not in the obs.])

== Look at

#note(c-flag)[Look at][
- `loss` on a batch from `sample` before any training: about 1 (the targets are standardized and the net outputs about 0).
- `act(obs)` has shape `(B, 16, 3)`.
- `list(policy.state_dict().keys())` shows `norm.obs_mean` and the `net.*` weights together.
]
] // claude:end
