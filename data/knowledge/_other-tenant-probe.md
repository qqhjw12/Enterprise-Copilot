---
title: 外租户测试文档（不应被其他租户检索到）
tenant: ten_other_demo
classification: internal
---

# 外租户测试文档

## 唯一标识

本段包含一个不会在其他文档出现的标识串：ZQ-OTHER-TENANT-9931。

## 内容

这份文档属于 ten_other_demo 租户，用于验证多租户隔离：任何非该租户的用户检索「ZQ-OTHER-TENANT-9931」时，都必须返回空结果。如果检索结果中出现了本文档，说明权限过滤失效，属于 P0 级数据泄漏缺陷。
