function Show(showid){
    const ShowID=showid;
    const QA = document.getElementsByName('QA');
    for(let n of QA){
        if(n.id==ShowID){
            if(n.className=='collapse show'){
                n.className='collapse';
            }else{
                n.className='collapse show';
            }
        }else{
            n.className='collapse';
        }
    }
    
}