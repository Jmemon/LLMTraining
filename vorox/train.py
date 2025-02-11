import torch


def train_step(model, optimizer, loss_fn, batch):
    model.train()
    optimizer.zero_grad()
    outputs = model(batch)
    loss = loss_fn(outputs, batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def epoch(model, train_loader, val_loader, optimizer, loss_fn):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            val_loss += loss.item()

    return train_loss / len(train_loader), val_loss / len(val_loader)


def fit(model, train_loader, val_loader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        train_loss, val_loss = epoch(model, train_loader, val_loader, optimizer, loss_fn)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    return model
