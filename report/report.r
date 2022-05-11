## ---- init ----

library(ggplot2)
library(hrbrthemes)
library(scales)
library(tidyverse)
library(pracma)
library(ggforce)
library(ggrepel)
library(reticulate)

theme_set(theme_ipsum(base_family = "") + theme(
  axis.title.x = element_text(hjust = 0.5),
  axis.title.y = element_text(hjust = 0.5), plot.margin = margin(
    t = 0.5,
    r = 2, b = 0.5, l = 2, "cm"
  )
))

knitr::opts_chunk$set(dev = "tikz", fig.align = "center")
options(tikzDefaultEngine = "luatex")

## ---- human ----

pi_scales <- math_format(.x, format = function(x) x / pi)

x_human_max <- 4
x_leg_index <- 10
x_human <- seq(1, x_human_max, 0.05) * pi

y_foot <- cospi(x_human / pi) + 1
y_knee <- cospi(x_human / pi) + 2

df_human <- data.frame(x = x_human, foot = y_foot, knee = y_knee)

g_human <- df_human %>%
  pivot_longer(-x) %>%
  ggplot(aes(x, value, colour = name)) +
  geom_line() +
  xlab("x") +
  ylab("y") +
  scale_x_continuous(labels = pi_scales, breaks = seq(0, x_human_max, 1) * pi)
g_human <- g_human + geom_segment(
  x = x_human[x_leg_index],
  y = y_foot[x_leg_index],
  xend = x_human[x_leg_index],
  yend = y_knee[x_leg_index],
  color = "orange"
)
g_human <- g_human + geom_segment(
  x = x_human[x_leg_index - 5],
  y = y_foot[x_leg_index - 5],
  xend = x_human[x_leg_index - 5],
  yend = y_knee[x_leg_index - 5],
  color = alpha("orange", 0.25),
  linetype = "longdash"
)

## ---- inter ----

x_inter <- 0:5
y_inter <- c(1, 0.5, 1, 0.75, 0.8, 2)

x_out_inter <- seq(0, tail(x_inter, 1), length.out = 1000)

df_smooth <- data.frame(approx(x_inter, y_inter, x_out_inter))
colnames(df_smooth) <- c("x", "linear")

df_smooth$cspline <- cubicspline(x_inter, y_inter, x_out_inter)

g_smooth <- df_smooth %>%
  pivot_longer(-x) %>%
  ggplot(aes(x, value, colour = name)) +
  geom_line()
g_smooth <- g_smooth +
  annotate("point", x = x_inter, y = y_inter, color = "orange", size = 3)

df_mono <- data.frame(x = x_out_inter)
df_mono$cspline <- df_smooth$cspline
df_mono$pchip <- pchip(x_inter, y_inter, x_out_inter)

g_mono <- df_mono %>%
  pivot_longer(-x) %>%
  ggplot(aes(x, value, colour = name)) +
  geom_line()
g_mono <- g_mono +
  annotate("point", x = x_inter, y = y_inter, color = "orange", size = 3)

## ---- angle ----

rotate_x_fun <- function(x, len, ang) {
  return(x - len * sinpi(ang))
}

rotate_y_fun <- function(y, len, ang) {
  return(y + len * cospi(ang))
}

directed_angle_fun <- function() {
  ang_pos <- 1 / 6
  ang_neg <- -1 / 6
  x <- c(0, 0)
  y <- c(0, 0)
  len <- 1
  xend <- rotate_x_fun(x, len, c(ang_pos, ang_neg))
  yend <- rotate_y_fun(y, len, c(ang_pos, ang_neg))
  xt <- rotate_x_fun(x, 0.6, c(ang_pos, ang_neg) / 2)
  yt <- rotate_y_fun(y, 0.6, c(ang_pos, ang_neg) / 2)

  df <- data.frame(
    x, y, xend, yend, xt, yt,
    label = c("+", "-"), ang = c(ang_neg, ang_pos)
  )

  g <- df %>% ggplot() +
    geom_segment(
      aes(x = x, y = y, xend = xend, yend = yend, color = label),
      size = 2
    ) +
    annotate(
      "segment",
      x = 0, y = 0, xend = 0, yend = 1, linetype = "dotted", size = 1
    ) +
    geom_arc(aes(
      x0 = x, y0 = y, r = 0.5,
      start = x, end = ang * pi, color = label
    ), linetype = "longdash", size = 1, arrow = arrow(
      angle = 30,
      ends = "last", type = "open"
    )) +
    geom_text(
      aes(
        x = xt, y = yt, label = paste0("$\\theta_{", label, "}$")
      )
    )

  return(g)
}

g_directed_angle <- directed_angle_fun()

## ---- robot ----

labs_robot <- factor(
  c("Body", "Upper Arm", "Lower Arm", "Atlatl", "Dart"),
  ordered = T,
  levels = rev(c("Body", "Upper Arm", "Lower Arm", "Atlatl", "Dart"))
)
labs_joint <- factor(
  c("Hip", "Shoulder", "Elbow", "Wrist", "P"),
  ordered = T,
  levels = rev(c("Hip", "Shoulder", "Elbow", "Wrist", "P"))
)

plot_robot_fun <- function(angs, lens, dot = 0.15, xlim = NULL, ylim = NULL) {
  seqs <- seq_along(angs)
  cumangs <- cumsum(angs)
  x <- c(0)
  y <- c(0)
  xend <- c()
  yend <- c()

  xdot <- c(0)
  ydot <- c(dot)

  xt <- c()
  yt <- c()

  for (i in seqs) {
    xi <- rotate_x_fun(x[i], lens[i], cumangs[i])
    yi <- rotate_y_fun(y[i], lens[i], cumangs[i])

    xt <- c(xt, rotate_x_fun(x[i], dot * 1.2, cumangs[i] - angs[i] / 2))
    yt <- c(yt, rotate_y_fun(y[i], dot * 1.2, cumangs[i] - angs[i] / 2))

    if (i != 1) {
      xdot <- c(xdot, rotate_x_fun(x[i], dot, cumangs[i - 1]))
      ydot <- c(ydot, rotate_y_fun(y[i], dot, cumangs[i - 1]))
    }

    xend <- c(xend, xi)
    yend <- c(yend, yi)
    x <- c(x, xi)
    y <- c(y, yi)
  }

  x <- head(x, -1)
  y <- head(y, -1)

  colors <- viridis::turbo(7)

  df <- data.frame(
    x,
    y,
    xend,
    yend,
    xdot,
    ydot,
    xt,
    yt,
    color = colors[seqs],
    label = labs_robot[seqs],
    joint = labs_joint[seqs],
    astart = -c(0, head(cumangs, -1)) * pi,
    aend = -cumangs * pi
  )

  g <- df %>% ggplot() +
    geom_segment(
      aes(x = x, y = y, xend = xend, yend = yend, color = label),
      size = 2
    ) +
    geom_segment(
      aes(x = x, y = y, xend = xdot, yend = ydot, color = label),
      linetype = "longdash", size = 1
    ) +
    geom_arc(aes(
      x0 = x, y0 = y, r = dot, start = astart, end = aend, color = label
    ), linetype = "longdash", size = 1) +
    scale_color_manual(values = df$color) +
    geom_label_repel(
      aes(x = x, y = y, label = joint),
      min.segment.length = 0,
      force_pull = 0.25
    ) +
    geom_text(aes(
      x = xt,
      y = yt,
      label = paste0("$\\mathbf{\\theta_", seqs, "}$")
    ))

  if (!is.null(xlim) && !is.null(ylim)) {
    g <- g + coord_cartesian(
      xlim = xlim, ylim = ylim, expand = F
    ) + theme(aspect.ratio = diff(ylim) / diff(xlim))
  }

  return(g)
}

## ---- initial-model ----

lens_robot <- c(0.603, 0.286, 0.279, sqrt(0.067**2 + 0.2**2), 0.5)
g_inital_robot <- plot_robot_fun(
  c(1 / 12, -1 / 10, 1 / 8, 3 / 5, -8 / 9),
  lens_robot
)

## ---- trajectory ----

source_python(here::here("report", "report.py"))
trajectory <- Trajectory()

times_traject <- cumsum(trajectory$times)
len_inter <- 1000
times_inter <- seq(
  from = times_traject[1],
  to = tail(times_traject, 1),
  length.out = len_inter
)

## ---- traject-hse ----

hse_traject <- trajectory$construct_angle_process()
hse_inter <- apply(
  hse_traject, 1, pchip,
  xi = times_traject, x = times_inter
)
df_hse <- data.frame(
  time = rep.int(times_inter, 3),
  ang = c(hse_inter),
  label = rep(labs_joint[1:3], each = len_inter)
)
g_hse <- df_hse %>% ggplot(aes(time, ang)) +
  geom_line() +
  facet_grid(label ~ .)

## ---- wrist-robot ----

plot_wrist_height_fun <- function(angs, lens) {
  seqs <- seq_along(angs)
  cumangs <- cumsum(angs)
  x <- c(0)
  y <- c(0)
  xend <- c()
  yend <- c()

  for (i in seqs) {
    xi <- rotate_x_fun(x[i], lens[i], cumangs[i])
    yi <- rotate_y_fun(y[i], lens[i], cumangs[i])

    xend <- c(xend, xi)
    yend <- c(yend, yi)
    x <- c(x, xi)
    y <- c(y, yi)
  }

  x <- head(x, -1)
  y <- head(y, -1)

  colors <- viridis::turbo(7)

  df <- data.frame(
    x,
    y,
    xend,
    yend,
    color = colors[seqs],
    label = labs_robot[seqs]
  )

  g <- df %>% ggplot() +
    geom_segment(
      aes(x = x, y = y, xend = xend, yend = yend, color = label),
      size = 2
    ) +
    geom_segment(
      x = df$xend[[4]], y = df$yend[[4]], xend = df$xend[[4]], yend = 0,
      color = df$color[[2]],
      linetype = "longdash", size = 0.5
    ) +
    geom_text(
      x = df$xend[[4]] + 0.025, y = df$yend[[4]] / 2,
      label = "h"
    ) +
    scale_color_manual(values = df$color)

  return(g)
}

g_wrist_robot <- plot_wrist_height_fun(
  c(1 / 12, -1 / 10, 1 / 8, 3 / 5, -8 / 9),
  lens_robot
)

## ---- traject-wrist ----

df_wrist <- data.frame(
  time = times_inter,
  ang = apply(hse_inter, 1, trajectory$calculate_last_angle)
)
g_wrist <- df_wrist %>% ggplot(aes(time, ang)) +
  geom_line()