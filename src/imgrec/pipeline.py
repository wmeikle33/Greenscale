from .metrics import accuracy

def run_train_eval(train_loader, test_loader, prep, model):
    ds_tr = train_loader.load()
    ds_te = test_loader.load()
    prep.fit(ds_tr)
    ds_tr = prep.transform(ds_tr)
    ds_te = prep.transform(ds_te)
    model.fit(ds_tr.X, ds_tr.y)
    y_pred = model.predict(ds_te.X)
    return {'accuracy': accuracy(ds_te.y, y_pred)}
